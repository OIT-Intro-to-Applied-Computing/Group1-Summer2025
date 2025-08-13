import argparse, pandas as pd, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=False)
    ap.add_argument("--video_root", required=False, default="datasets/training_data") # Default to datasets/training_data
    ap.add_argument("--pose_root", required=False, default="datasets/training_data")
    ap.add_argument("--outdir", default="data/manifests")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    def make(tsv_in, split):
        # Always read with tab separator, as filtered CSVs are tab-separated
        df = pd.read_csv(tsv_in, sep="\t")
        # Expect columns: SENTENCE_NAME, SENTENCE, VIDEO_PATH (or build from roots)
        
        # Ensure VIDEO_PATH column exists, even if empty
        if "VIDEO_PATH" not in df.columns:
            df["VIDEO_PATH"] = "" # Initialize with empty string

        if "SENTENCE_NAME" in df.columns and args.video_root:
            # Construct VIDEO_PATH based on split and SENTENCE_NAME
            # Corrected path to include intermediate directories
            df["VIDEO_PATH"] = df["SENTENCE_NAME"].apply(lambda x: os.path.join(args.video_root, f"{split}_rgb_front_clips/raw_videos/{x}.mp4"))
        
        # Ensure POSE_DIR column exists, even if empty
        if "POSE_DIR" not in df.columns:
            df["POSE_DIR"] = "" # Initialize with empty string

        if "SENTENCE_NAME" in df.columns and args.pose_root:
            df["POSE_DIR"] = df["SENTENCE_NAME"].apply(lambda x: os.path.join(args.pose_root, f"{split}_2D_keypoints/openpose_output/json/{x}"))

        out = os.path.join(args.outdir, f"{split}.tsv")
        df[["SENTENCE_NAME","VIDEO_PATH","SENTENCE","POSE_DIR"]].to_csv(out, sep="\t", index=False)
        print("Wrote", out)

    make(args.train, "train")
    make(args.val, "val")
    if args.test:
        make(args.test, "test")

if __name__ == "__main__":
    main()
