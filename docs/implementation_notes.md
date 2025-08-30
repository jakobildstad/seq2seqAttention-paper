# Implementation notes
seq2seq en-no translation with attention

## Dataset preprocessing and tokenization

A joint shared **subword** vocab (tokenizer: SentencePiece BPE or Unigram) trianed on both languages. 

### Sentencepiece
SentencePiece is a tokenizer that breaks raw text into subword units (Byte-Pair Encoding or Unigram). It’s designed for NLP training pipelines:
- It treats the input as a raw stream of Unicode (no need to pre-segment by spaces).
- It learns a vocabulary of pieces (like “▁Hotel”, “s”, “▁Mc”, “Cleary”) to cover your dataset efficiently.
- It produces both a .model file (binary, used for encoding/decoding) and a .vocab file (human-readable list of pieces).

The special tokens (PAD, UNK, BOS, EOS) are reserved tokens with fixed IDs so your model knows where sequences start, end, and how to handle padding.
´´´
SPECIALS = {
    "pad_id": 0, "pad_piece": "<pad>",
    "unk_id": 1, "unk_piece": "<unk>",
    "bos_id": 2, "bos_piece": "<bos>",
    "eos_id": 3, "eos_piece": "<eos>",
}
´´´
- pad_piece: Padding token. Used to pad all sequences in a batch to the same length. Your model ignores this in loss calculations.
- unk_piece: Unknown token. Replaces any character sequence the tokenizer didn’t see during training (rare with subwords).
- bos_piece: Beginning of sequence token. Always added at the start of a sequence so the decoder knows when to start.
- eos_piece: End of sequence token. Marks the end of a sequence. The decoder stops generating after this.

- encode (script): text → subword pieces/ids.
- decode (script): subword ids → text (detokenization).

### Corpus building

Takes your TSV file (en-nb.txt with English and Norwegian separated by a tab).
	•	For each line, splits into English en and Norwegian nb.
	•	Writes each separately to a single temporary text file.
SentencePiece expects one sentence per line, so this doubles your dataset (every translation pair contributes two lines).