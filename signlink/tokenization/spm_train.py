import argparse, pandas as pd, os, sentencepiece as spm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="TSV path")
    ap.add_argument("--text-col", default="SENTENCE")
    ap.add_argument("--vocab_size", type=int, default=12000)
    ap.add_argument("--out", default="artifacts/spm")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = pd.read_csv(args.input, sep="\t")
    txt = os.path.join(os.path.dirname(args.out), "corpus.txt")
    df[args.text_col].to_csv(txt, index=False, header=False)

    spm.SentencePieceTrainer.Train(
        input=txt,
        model_prefix=args.out,
        vocab_size=args.vocab_size,
        character_coverage=1.0,
        model_type="bpe",
        pad_id=0, bos_id=1, eos_id=2, unk_id=3
    )
    print("Saved:", args.out + ".model")

if __name__ == "__main__":
    main()
