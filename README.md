# SignLink (1D-CNN + Transformer)

[📄 **View Lab Research Paper**](https://github.com/OIT-Intro-to-Applied-Computing/Group1-Summer2025/blob/main/OIT-%20Sign%20Link_%20A%201d-cnn%20%2B%20Transformer%20Approach%20To%20Sign%20Language%20Translation.pdf)

> **Goal:** Translate sign language videos → English text using deep learning.

## Core Components

* **Per-frame CNN (ResNet18)** → 512-d embeddings
* **Temporal 1D-CNN** → downsample & model motion
* *(Optional)* Tiny Transformer encoder
* **Transformer decoder** (seq2seq) → text (SentencePiece)
* **Pose data integration** for better accuracy

## Overview

SignLink is a Sign Language Translation system combining 1D-CNN temporal modeling with Transformer decoding, integrating pose landmarks for improved translation quality. It supports training, evaluation, and real-time inference via a Streamlit app.

## Key Features

* **Video Encoding:** ResNet18 frame-wise features
* **Temporal Modeling:** 1D-CNN for sequence compression
* **Pose Fusion:** 2D keypoints integrated with temporal features
* **Transformer Decoder:** Attention-based translation
* **SentencePiece Tokenization:** Subword processing
* **Streamlit App:** Interactive inference & evaluation

## Requirements

* Python 3.9+
* PyTorch
* SentencePiece
* Decord *(optional)* for faster video loading
* OpenCV, Pandas, Streamlit
* Other dependencies in `requirements.txt`

```bash
pip install -r requirements.txt
```

## Data Preparation

1. **Download Videos:** Get the How2Sign dataset from the official site → [Download Dataset](https://how2sign.github.io/)
2. **Generate Keypoints:** Use OpenPose or similar.
3. **Filter CSV Files:**

```bash
python datasets/training_data/filter_csv.py
```

4. **Build Manifest Files:**

```bash
python scripts/build_manifests.py \
   --train datasets/training_data/how2sign_realigned_train_filtered.csv \
   --val datasets/training_data/how2sign_realigned_val_filtered.csv \
   --test datasets/training_data/how2sign_realigned_test_filtered.csv \
   --pose_root datasets/training_data \
   --video_root datasets/training_data \
   --outdir data/manifests
```

## Training

### Train SentencePiece

```bash
python -m signlink.tokenization.spm_train \
    --input data/manifests/train.tsv \
    --text-col SENTENCE \
    --vocab_size 12000 \
    --out artifacts/spm
```

### Train Model

```bash
python -m signlink.train.finetune_slt \
    --config configs/slt_temporal1d_base.yaml
```

## Evaluation

```bash
python -m signlink.train.evaluate \
    --manifest data/manifests/test.tsv \
    --ckpt checkpoints/slt_best.pth \
    --config configs/slt_temporal1d_base.yaml
```

## Inference with Streamlit App

```bash
PYTHONPATH=$PYTHONPATH:/home/amdal/Developer/SignLink_1dcnn_transformer \
streamlit run streamlit_app/app.py
```

* Provide checkpoint, SentencePiece model, and pose root path
* Upload video + sentence name → get translation
* Evaluate via manifest TSV upload

## Model Architecture

* **FrameCNN:** ResNet18
* **Temporal1DEncoder:** Temporal CNN layers
* **TextDecoderTFM:** Transformer decoder

## Configuration

Settings are in `configs/slt_temporal1d_base.yaml`.

## Data Format

```
SENTENCE_NAME\tVIDEO_PATH\tSENTENCE\tPOSE_DIR
```

## Contact

Ahmed Ali (*LordAmdal*) — [amdal.ali@oit.edu](mailto:amdal.ali@oit.edu)
