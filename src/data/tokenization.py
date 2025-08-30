# tok_tsv_only.py
# Joint SentencePiece tokenizer for EN<->NO where input is a TSV:
# each line: "<english>\t<norwegian>"
#
# Train:
#   python tok_tsv_only.py train \
#       --tsv data/en-nb.txt \
#       --out data/spm_joint \
#       --vocab_size 8000 --model_type bpe
#
# Encode:
#   python tok_tsv_only.py encode \
#       --spm data/spm_joint.model \
#       --tsv data/en-nb.txt \
#       --out_src data/train.en.spm --out_tgt data/train.no.spm \
#       --out_src_ids data/train.en.ids --out_tgt_ids data/train.no.ids
#
# Decode (ids -> text, one column):
#   python tok_tsv_only.py decode \
#       --spm data/spm_joint.model \
#       --inp_ids data/pred.ids \
#       --out data/pred.txt

import argparse, os, io, sentencepiece as spm

SPECIALS = {
    "pad_id": 0, "pad_piece": "<pad>",
    "unk_id": 1, "unk_piece": "<unk>",
    "bos_id": 2, "bos_piece": "<bos>",
    "eos_id": 3, "eos_piece": "<eos>",
}

def _concat_corpus_from_tsv(tmp_path, tsv_path):
    n, kept = 0, 0
    with io.open(tmp_path, "w", encoding="utf-8") as w, \
         io.open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            n += 1
            s = line.rstrip("\n")
            if not s: 
                continue
            # split once; anything after first tab belongs to NO
            parts = s.split("\t", 1)
            if len(parts) != 2:
                continue  # skip malformed lines
            en, nb = parts[0].strip(), parts[1].strip()
            if en:
                w.write(en + "\n")
                kept += 1
            if nb:
                w.write(nb + "\n")
                kept += 1
    print(f"[train] read {n} TSV lines; wrote {kept} mono lines to temp corpus")

def train(args):
    tmp_corpus = args.out + ".train.txt"
    _concat_corpus_from_tsv(tmp_corpus, args.tsv)

    spm.SentencePieceTrainer.Train(
        input=tmp_corpus,
        model_prefix=args.out,                 # -> out.model, out.vocab
        vocab_size=args.vocab_size,
        model_type=args.model_type,            # "bpe" or "unigram"
        character_coverage=1.0,                # Latin scripts
        input_sentence_size=args.input_sentence_size,  # 0 = use all lines
        shuffle_input_sentence=True,
        pad_id= SPECIALS["pad_id"], pad_piece= SPECIALS["pad_piece"],
        unk_id= SPECIALS["unk_id"], unk_piece= SPECIALS["unk_piece"],
        bos_id= SPECIALS["bos_id"], bos_piece= SPECIALS["bos_piece"],
        eos_id= SPECIALS["eos_id"], eos_piece= SPECIALS["eos_piece"],
        user_defined_symbols="",
        normalization_rule_name="nfkc",
        train_extremely_large_corpus=False,
    )
    os.remove(tmp_corpus)
    print(f"Trained: {args.out}.model  and  {args.out}.vocab")

def _load(spm_path):
    sp = spm.SentencePieceProcessor()
    if not sp.Load(spm_path):
        raise SystemExit(f"Failed to load SentencePiece model: {spm_path}")
    return sp

def _encode_one(text, sp, wp, wi):
    pieces = [SPECIALS["bos_piece"]] + sp.EncodeAsPieces(text) + [SPECIALS["eos_piece"]]
    ids    = [SPECIALS["bos_id"]]    + sp.EncodeAsIds(text)    + [SPECIALS["eos_id"]]
    wp.write(" ".join(pieces) + "\n")
    wi.write(" ".join(map(str, ids)) + "\n")

def encode(args):
    sp = _load(args.spm)
    lines, kept = 0, 0
    with io.open(args.tsv, "r", encoding="utf-8") as fin, \
         io.open(args.out_src, "w", encoding="utf-8") as wp_src, \
         io.open(args.out_tgt, "w", encoding="utf-8") as wp_tgt, \
         io.open(args.out_src_ids, "w", encoding="utf-8") as wi_src, \
         io.open(args.out_tgt_ids, "w", encoding="utf-8") as wi_tgt:
        for raw in fin:
            lines += 1
            s = raw.rstrip("\n")
            if not s:
                continue
            parts = s.split("\t", 1)
            if len(parts) != 2:
                continue
            en, nb = parts[0].strip(), parts[1].strip()
            if en:
                _encode_one(en, sp, wp_src, wi_src); kept += 1
            else:
                wp_src.write("\n"); wi_src.write("\n")
            if nb:
                _encode_one(nb, sp, wp_tgt, wi_tgt); kept += 1
            else:
                wp_tgt.write("\n"); wi_tgt.write("\n")

    print(f"[encode] read {lines} TSV lines; encoded {kept} sequences")
    print("Encoded pieces ->", args.out_src, "and", args.out_tgt)
    print("Encoded ids    ->", args.out_src_ids, "and", args.out_tgt_ids)

def decode(args):
    sp = _load(args.spm)
    with io.open(args.inp_ids, "r", encoding="utf-8") as f, \
         io.open(args.out, "w", encoding="utf-8") as w:
        for line in f:
            toks = line.strip().split()
            if not toks:
                w.write("\n"); continue
            ids = [int(x) for x in toks]
            if ids and ids[0] == SPECIALS["bos_id"]: ids = ids[1:]
            if ids and ids[-1] == SPECIALS["eos_id"]: ids = ids[:-1]
            w.write(sp.DecodeIds(ids) + "\n")
    print("Decoded ->", args.out)

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train")
    pt.add_argument("--tsv", required=True, help="TSV with EN\\tNO per line")
    pt.add_argument("--out", required=True, help="SentencePiece prefix, e.g., data/spm_joint")
    pt.add_argument("--vocab_size", type=int, default=8000)
    pt.add_argument("--model_type", choices=["bpe","unigram"], default="bpe")
    pt.add_argument("--input_sentence_size", type=int, default=0, help="0=use all; else subsample this many lines")
    pt.set_defaults(func=train)

    pe = sub.add_parser("encode")
    pe.add_argument("--spm", required=True)
    pe.add_argument("--tsv", required=True, help="TSV with EN\\tNO per line")
    pe.add_argument("--out_src", required=True)
    pe.add_argument("--out_tgt", required=True)
    pe.add_argument("--out_src_ids", required=True)
    pe.add_argument("--out_tgt_ids", required=True)
    pe.set_defaults(func=encode)

    pd = sub.add_parser("decode")
    pd.add_argument("--spm", required=True)
    pd.add_argument("--inp_ids", required=True)
    pd.add_argument("--out", required=True)
    pd.set_defaults(func=decode)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()