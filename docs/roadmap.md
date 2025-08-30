# Seq2Seq with Attention - Implementation Roadmap

## Phase 0 — Ground rules (repo + reproducibility)
- Create a clean repo: seq2seq/ with data/ src/ scripts/ tests/ runs/.
- Pin versions; set seeds; enable deterministic cuDNN if needed.
- YAML config: model dims, optimizer, scheduler, data paths, training hyperparams.
- Logging: TensorBoard or Weights & Biases; save checkpoints + config + git hash.

**Deliverable**: `python -m src.train --config configs/base.yaml` runs and logs.

## Phase 1 — Data pipeline
1. **Dataset**: start with a small parallel corpus (e.g., toy EN–FR, or your own paired lines).
2. **Tokenization**:
   - Use subwords (BPE/Unigram) via SentencePiece; target vocab_size≈8k–16k.
   - Learn two vocabularies or a shared one; add `<pad>` `<bos>` `<eos>` `<unk>`.
3. **Numericalization**:
   - Map to ids; truncate/clip long sentences for the first run (e.g., ≤50 tokens).
4. **Batched DataLoader**:
   - Bucket by length; pad to batch max; return attention masks.

**Sanity checks**:
- Print a batch: shapes (B, T_src), (B, T_tgt), masks; ensure `<bos>`/`<eos>` placement is correct.
- Verify reversible detokenization on a couple samples.

## Phase 2 — Minimal vanilla Seq2Seq (no attention)

**Components in src/models/**:
- **Encoder**: BiGRU (or BiLSTM). Inputs: src_ids, src_mask. Outputs:
  - H = contextual states (B, T_src, 2*Henc).
  - Optional summary h_summary (concat last forward + first backward).
- **Decoder**: GRU with input concat(emb(y_{i-1}), c_i) but for now let c_i = linear(h_summary).
- **Generator**: Linear([s_i; c_i] → vocab_logits).

**Training loop (teacher forcing)**:
- Shift targets right: y_in = [`<bos>`, y_0..y_{N-1}], y_out = [y_0..y_N].
- Loss: cross-entropy on y_out with ignore_index=`<pad>`.
- Optimizer: Adam (β1=0.9, β2=0.98) lr=3e-4; gradient clip=1.0.
- Schedule: warmup 4k steps or cosine; not critical for first run.

**Sanity checks**:
- Overfit 100 sentence pairs to near-0 loss.
- Greedy decode a line; see recognizable output.

## Phase 3 — Add Bahdanau attention ("alignment model")

**Add module**:
- score(s_prev, h_j) = v_a^T tanh(W_a s_prev + U_a h_j) where
  - s_prev: (B, Hdec), h_j: (B, T_src, Henc2).
  - Broadcast W_a s_prev over T_src.
  - Masked softmax over j using src_mask to obtain α_{ij}.
  - c_i = Σ_j α_{ij} h_j (batch matmul).
- Decoder step: s_i = GRU([emb(y_{i-1}); c_i], s_{i-1}).
- Output: logits_i = W_o [s_i; c_i] + b.

**Implementation notes**:
- Keep all operations 2D/3D; avoid Python loops if you later add teacher forcing with parallel time (you can also implement step-wise decoding with a for-loop first).
- Apply mask before softmax: set padded positions to -1e9.

**Sanity checks**:
- Attention weights sum to 1 across source length (on non-padded positions).
- Visualize α heatmaps for a few pairs; see diagonalish patterns.

## Phase 4 — Training at scale
- Batch size: start 64 (tokens per batch limit is better: e.g., 4k tokens).
- Regularization: dropout 0.1–0.3 on embeddings, RNN outputs, and in attention MLP.
- Label smoothing 0.1 can help.
- Mixed precision (fp16/bf16) with loss scaling.

**Validation**:
- Compute token-level loss on dev.
- Text metric: BLEU (sacreBLEU) or chrF. Start with BLEU on detok outputs.

**Early stopping**:
- Monitor dev BLEU and dev loss; patience 3–5 evals.

## Phase 5 — Inference
- **Greedy decoding**: simple baseline.
- **Beam search**:
  - Beam size 4–8.
  - Length penalty lp = ((5+len)^α)/(6^α) with α≈0.6.
  - Stop when all beams emit `<eos>` or hit max length.
  - Optional: coverage penalty (light) to discourage repetitions.

**Sanity checks**:
- Greedy vs beam improvements on a few dev sentences.
- Ensure no infinite loops; enforce max length (e.g., 1.5×src_len + 10).

## Phase 6 — Tests (prevent silent shape bugs)

**Unit tests (pytest) you can write quickly**:
- `test_attention_masking.py`: masked positions receive ~0 prob after softmax.
- `test_seq_lengths.py`: encoder correctly handles varying lengths; H.shape.
- `test_step_vs_parallel_decoder.py`: step-wise teacher forcing logits match time-parallel version.
- `test_save_load.py`: checkpoint round-trip gives identical outputs.

## Phase 7 — Quality & speed knobs
- Swap GRU↔LSTM; increase hidden sizes (e.g., Enc 512×2, Dec 512).
- Tie weights: share decoder embedding and output projection (transpose).
- Pretrained embeddings (optional; subwords make this less useful).
- Better batching: dynamic bucketing by length; tokens-per-batch.
- Gradient accumulation to simulate larger batches.

## Phase 8 — Observability & debugging heuristics
- Plot train/dev loss; watch for divergence after adding attention → usually masking bug.
- Inspect attention maps; look for diffuse vs peaky behavior.
- Track % `<unk>` and average decoded length; wildly short/long outputs signal length penalty or `<eos>` handling issues.
- Check exposure bias by sampling with low temperature at training checkpoints.

## Phase 9 — Packaging & CLI
- `scripts/train.py` with config path.
- `scripts/translate.py --ckpt path --src file.txt --out out.txt --beam 5`.
- Save tokenizer model + vocab with the checkpoint; ensure deterministic decode.

## Phase 10 — Stretch goals
- Multi-layer encoder/decoder with residuals.
- Coverage attention or input feeding (Luong) for stability.
- Byte-level BPE (shared vocab).
- Mixed-task pretraining (denoising) to improve low-data performance.
- Port to a Transformer to compare BLEU/EER-style curves on length.
