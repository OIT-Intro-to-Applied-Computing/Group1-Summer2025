# SignLink (1D-CNN + Transformer)

[Link to Research Paper](OIT- Sign Link_ A 1d-cnn + Transformer Approach To Sign Language Translation.pdf)


Sign Language Translation (video → English) using:
- Per-frame CNN (ResNet18) → 512-d embeddings
- Temporal 1D-CNN (downsamples time)
- Optional tiny Transformer encoder
- Transformer decoder (seq2seq) to text (SentencePiece)
- Pose data integration

## Overview

This project implements a sign language translation system using a 1D-CNN and Transformer architecture. It takes video and pose data as input and translates it into English text.

## Key Features

- **Video Encoding**: Uses a per-frame CNN (ResNet18) to extract visual features from video frames.
- **Temporal Modeling**: Employs a temporal 1D-CNN to model the temporal dynamics of the video.
- **Pose Integration**: Integrates pose data (2D keypoints) to enhance translation accuracy.
- **Transformer Decoder**: Uses a Transformer decoder to generate the English translation.
- **SentencePiece Tokenization**: Utilizes SentencePiece for subword tokenization of the output text.
- **Streamlit Inference App**: Provides a user-friendly Streamlit application for performing inference on uploaded videos.

## Requirements

- Python 3.9+
- PyTorch
- SentencePiece
- Decord (optional, but recommended for faster video loading)
- OpenCV
- Pandas
- Streamlit
- Other dependencies listed in `requirements.txt`

To install the dependencies, run:

```bash
pip install -r requirements.txt
```

## Data Preparation

1.  **Download Videos**: Download the sign language videos for your dataset.
2.  **Generate Keypoints**: Use OpenPose or a similar tool to generate 2D keypoints for each video frame. Ensure the keypoint data is stored in the correct directory structure (e.g., `datasets/training_data/<split>_2D_keypoints/openpose_output/json/<SENTENCE_NAME>/`).
3.  **Filter CSV Files**: Use the `filter_csv.py` script to filter your CSV files based on the existence of keypoint data:

    ```bash
    python datasets/training_data/filter_csv.py
    ```

4.  **Build Manifest Files**: Use the `build_manifests.py` script to generate the manifest files (train.tsv, val.tsv, test.tsv) that the data loaders will use. Provide the correct paths to your training, validation, and test CSV files, as well as the root directory for your video and pose data:

    ```bash
    python scripts/build_manifests.py --train datasets/training_data/how2sign_realigned_train_filtered.csv --val datasets/training_data/how2sign_realigned_val_filtered.csv --test datasets/training_data/how2sign_realigned_test_filtered.csv --pose_root datasets/training_data --video_root datasets/training_data --outdir data/manifests
    ```

## Training

1.  **Train SentencePiece Model**: Train a SentencePiece model for tokenization:

    ```bash
    python -m signlink.tokenization.spm_train --input data/manifests/train.tsv --text-col SENTENCE --vocab_size 12000 --out artifacts/spm
    ```

2.  **Train the Sign2Text Model**: Use the `finetune_slt.py` script to train the model. Make sure to specify the correct configuration file:

    ```bash
    python -m signlink.train.finetune_slt --config configs/slt_temporal1d_base.yaml
    ```

    The training process will log metrics (training loss, validation loss, BLEU, chrF) to a `training_log.csv` file in your project's root directory.

## Evaluation

Use the `evaluate.py` script to evaluate the trained model on the validation or test set:

```bash
python -m signlink.train.evaluate --manifest data/manifests/test.tsv --ckpt checkpoints/slt_best.pth --config configs/slt_temporal1d_base.yaml
```

This will print evaluation metrics (BLEU, chrF, prediction length statistics) to the console and save predictions to `outputs/preds.csv`.

## Inference with Streamlit App

1.  **Run the Streamlit App**: Use the following command to launch the Streamlit application:

    ```bash
    PYTHONPATH=$PYTHONPATH:/home/amdal/Developer/SignLink_1dcnn_transformer streamlit run streamlit_app/app.py
    ```

2.  **Using the App**:
    *   In the Streamlit sidebar, specify the paths to your checkpoint file (`checkpoints/slt_best.pth`), SentencePiece model (`artifacts/spm.model`), and pose data root directory (`datasets/training_data`).
    *   In the "Demo" tab, upload a video file and enter the corresponding `SENTENCE_NAME` for the video. The app will then display the predicted translation.
    *   In the "Evaluate" tab, upload a manifest TSV file to evaluate the model on a set of videos.

## Model Architecture

The core model architecture is defined in `signlink/models/sign2text_model.py` and consists of:

-   **FrameCNN**: A per-frame CNN (ResNet18) to extract visual features.
-   **Temporal1DEncoder**: A temporal 1D-CNN to model the temporal dynamics of the video.
-   **TextDecoderTFM**: A Transformer decoder to generate the English translation.

## Configuration

The training and model parameters are configured in `configs/slt_temporal1d_base.yaml`.

## Data Format

The training and validation data are specified in TSV files with the following schema:

```
SENTENCE_NAME\tVIDEO_PATH\tSENTENCE\tPOSE_DIR
```

-   `SENTENCE_NAME`: A unique identifier for the video.
-   `VIDEO_PATH`: The path to the video file.
-   `SENTENCE`: The English translation of the sign language in the video.
-   `POSE_DIR`: The path to the directory containing the pose data (2D keypoints) for the video. This directory should contain JSON files, one for each frame of the video.

## Contact

For questions or contributions, please contact @LordAmdal at amdal.ali@oit.edu.
