

import pandas as pd
import os

def filter_csv_by_existing_keypoints(csv_path, keypoints_base_dir, output_csv_path):
    """
    Filters a CSV file to only include entries for which keypoint data exists.

    Args:
        csv_path (str): The path to the input CSV file.
        keypoints_base_dir (str): The path to the directory containing the keypoint directories.
        output_csv_path (str): The path to save the filtered CSV file.
    """
    # 1. Get the list of all the directory names in the keypoints directory
    try:
        existing_keypoint_dirs = [d for d in os.listdir(keypoints_base_dir) if os.path.isdir(os.path.join(keypoints_base_dir, d))]
    except FileNotFoundError:
        print(f"Error: Directory not found at {keypoints_base_dir}")
        return

    # 2. Read the CSV file into a pandas DataFrame
    try:
        df = pd.read_csv(csv_path, sep='\t')
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # 3. Filter the DataFrame
    filtered_df = df[df['SENTENCE_NAME'].isin(existing_keypoint_dirs)]

    # 4. Save the filtered DataFrame to a new CSV file
    filtered_df.to_csv(output_csv_path, sep='\t', index=False)
    print(f"Filtered CSV saved to {output_csv_path}")
    print(f"Original number of entries: {len(df)}")
    print(f"Number of entries with existing keypoints: {len(filtered_df)}")

if __name__ == '__main__':
    # For training data
    train_csv_path = 'how2sign_realigned_train.csv'
    train_keypoints_dir = 'train_2D_keypoints/openpose_output/json'
    train_output_csv_path = 'how2sign_realigned_train_filtered.csv'
    filter_csv_by_existing_keypoints(train_csv_path, train_keypoints_dir, train_output_csv_path);

    # For validation data
    val_csv_path = 'how2sign_realigned_val.csv'
    val_keypoints_dir = 'val_2D_keypoints/openpose_output/json'
    val_output_csv_path = 'how2sign_realigned_val_filtered.csv'
    filter_csv_by_existing_keypoints(val_csv_path, val_keypoints_dir, val_output_csv_path);

    # For test data
    test_csv_path = 'how2sign_realigned_test.csv'
    test_keypoints_dir = 'test_2D_keypoints/openpose_output/json'
    test_output_csv_path = 'how2sign_realigned_test_filtered.csv'
    filter_csv_by_existing_keypoints(test_csv_path, test_keypoints_dir, test_output_csv_path)

