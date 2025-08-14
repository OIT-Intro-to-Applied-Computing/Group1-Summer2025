# From Data to Training: Preparing ASL Data for a 1D-CNN + Transformer Model - OIT Applied Computing

This document outlines the process of transforming the How2Sign dataset, including its CSV files and keypoint data, into a suitable format for training a 1D-CNN + Transformer network for sign language translation.

## 1. Data Organization

The dataset is organized into two main components:

- **Metadata (CSV Files):** The `how2sign_realigned_train.csv`, `how2sign_realigned_val.csv`, and `how2sign_realigned_test.csv` files serve as the master index. Each row in these files links a specific video segment to its English translation.

  - `SENTENCE_NAME`: This column is a unique identifier for a video segment (e.g., `--7E2sU6zP4_10-5-rgb_front`).
  - `SENTENCE`: This column contains the corresponding ground-truth English sentence.

- **Keypoint Data (JSON Files):** The 2D keypoint information, extracted using OpenPose, is stored in a nested directory structure. For each video segment, there is a corresponding directory named after its `SENTENCE_NAME` located in `train_2D_keypoints/openpose_output/json/`, `val_2D_keypoints/openpose_output/json/`, etc. Each of these directories contains a sequence of JSON files, where each file represents the keypoints for a single frame of the video.

## 2. Data Processing Pipeline

To train our network, we need to process this raw data into numerical tensors. The pipeline involves the following steps:

### Step 1: Read Metadata

We start by loading the CSV file into a pandas DataFrame to easily access the `SENTENCE_NAME` and `SENTENCE` for each training example.

### Step 2: Locate and Load Keypoint Data

For each row in the DataFrame, we use the `SENTENCE_NAME` to construct the path to the directory containing the sequence of keypoint JSON files. We then read and parse each JSON file in chronological order.

### Step 3: Extract and Normalize Keypoints

Each JSON file contains the 2D coordinates `(x, y)` and a confidence score `c` for various body joints. We will extract the `pose_keypoints_2d` for each person detected in the frame.

To ensure the model is robust to variations in signer position and scale, we must normalize the keypoints. A common technique is to make the keypoints relative to a central point and scale them.

**Equation for Normalization:**

Let \(P_{t,i} = (x_{t,i}, y_{t,i})\) be the coordinate of the \(i^{th}\) keypoint in frame \(t\). We select a reference keypoint, such as the neck (usually keypoint index 1 in OpenPose), denoted as \(P_{t,ref} = (x_{t,ref}, y_{t,ref})\).

The translation-normalized keypoint \(P'_{t,i}\) is:

\[ P'_{t,i} = P_{t,i} - P_{t,ref} \]

This centers the signer in the frame. After normalization, we flatten the `(x, y)` coordinates for all keypoints in a single frame into a feature vector. For \(N\) keypoints, the feature vector \(v_t\) for frame \(t\) will be:

\[ v_t = [P'_{t,1,x}, P'_{t,1,y}, P'_{t,2,x}, P'_{t,2,y}, ..., P'_{t,N,x}, P'_{t,N,y}] \]

### Step 4: Create Feature Tensors

The sequence of feature vectors \((v_1, v_2, ..., v_T))\) for all \(T\) frames of a video segment is stacked to create a 2D tensor of shape `(T, 2*N)`, where `T` is the number of frames and `N` is the number of keypoints.

This tensor, representing the spatio-temporal dynamics of the sign, serves as the primary input to our neural network.

## 3. Model Architecture: 1D-CNN + Transformer

Our network combines a 1D Convolutional Neural Network (CNN) with a Transformer encoder:

1.  **1D-CNN:** The input tensor `(T, 2*N)` is fed into a 1D-CNN. The convolutional filters slide along the temporal dimension (`T`) to learn local motion patterns and extract low-level features from the sequence of keypoints.

2.  **Transformer Encoder:** The sequence of features extracted by the CNN is then passed to a Transformer encoder. The self-attention mechanism within the Transformer excels at capturing long-range dependencies and contextual relationships between different parts of the signing sequence, producing a rich, context-aware representation.

3.  **Output Layer:** Finally, the output from the Transformer is passed to a linear layer followed by a softmax function to predict the probability distribution over the target vocabulary (English words or sub-words).

## 4. Sample Code

Here is a simplified Python snippet illustrating the data loading and preprocessing steps:

```python
import pandas as pd
import numpy as np
import os
import json

def load_and_process_keypoints(keypoint_dir):
    """Loads, normalizes, and stacks keypoints from a directory of JSON files."""
    if not os.path.exists(keypoint_dir) or not os.path.isdir(keypoint_dir):
        return None

    json_files = sorted([f for f in os.listdir(keypoint_dir) if f.endswith('.json')])
    frame_keypoints = []

    for file_name in json_files:
        with open(os.path.join(keypoint_dir, file_name), 'r') as f:
            data = json.load(f)

        # Assuming one person, extract pose keypoints (x, y, confidence)
        if data['people']:
            keypoints = np.array(data['people'][0]['pose_keypoints_2d']).reshape(-1, 3)
            coords = keypoints[:, :2]  # Extract (x, y) coordinates

            # Normalize relative to the neck keypoint (index 1)
            if coords.shape[0] > 1:
                neck = coords[1]
                normalized_coords = coords - neck
                frame_keypoints.append(normalized_coords.flatten()) # Flatten to a 1D vector

    if not frame_keypoints:
        return None

    return np.stack(frame_keypoints)

# 1. Load the metadata
csv_path = 'how2sign_realigned_train.csv'
df = pd.read_csv(csv_path, sep='\t')

# 2. Get a sample entry
sample_row = df.iloc[10]
sentence_name = sample_row['SENTENCE_NAME']
english_sentence = sample_row['SENTENCE']

# 3. Construct path and process keypoints
keypoints_base_dir = 'train_2D_keypoints/openpose_output/json'
keypoint_dir_path = os.path.join(keypoints_base_dir, sentence_name)

print(f"Processing: {sentence_name}")
print(f"Translation: {english_sentence}")

# This tensor is the input for a single training example
input_tensor = load_and_process_keypoints(keypoint_dir_path)

if input_tensor is not None:
    print(f"Generated Input Tensor Shape: {input_tensor.shape}")
else:
    print("Could not process keypoints for this sample.")

```

```

## Team Cortex 
@lordAmdal - Ahmed Ali  