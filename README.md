# SignLink (1D-CNN + Transformer) — MVP

Lightweight Sign Language Translation (video → English) using:
- Per-frame CNN (ResNet18) → 512-d embeddings
- Temporal 1D-CNN (downsamples time)
- Optional tiny Transformer encoder
- Transformer decoder (seq2seq) to text (SentencePiece)

## Quickstart

```bash
python -m signlink.tokenization.spm_train --input data/manifests/train.tsv --text-col SENTENCE --vocab_size 12000 --out artifacts/spm

python -m signlink.train.finetune_slt --config configs/slt_temporal1d_base.yaml

python -m signlink.train.evaluate --manifest data/manifests/val.tsv --ckpt checkpoints/slt_best.pth

streamlit run streamlit_app/app.py
```

Input TSV schema: `SENTENCE_NAME\tVIDEO_PATH\tSENTENCE\tPOSE_DIR`
`POSE_DIR` may be empty/omitted.
# Group1-Summer2025
