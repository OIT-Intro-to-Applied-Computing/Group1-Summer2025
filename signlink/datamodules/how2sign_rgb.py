import torch, pandas as pd
from torch.utils.data import Dataset
from ..utils.video_io import load_clip_frames
from ..utils.pose_io import load_pose_or_none # New import

class How2SignRGB(Dataset):
    def __init__(self, tsv_path, img_size=192, frames=32, fps=25, frame_stride=2, frame_drop_rate=0.0): # Added frame_drop_rate
        self.df = pd.read_csv(tsv_path, sep="\t")
        self.img_size = img_size; self.frames = frames; self.fps=fps; self.frame_stride=frame_stride; self.frame_drop_rate=frame_drop_rate # Store frame_drop_rate
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video = str(row["VIDEO_PATH"]); text = str(row["SENTENCE"])
        pose_dir = str(row["POSE_DIR"]) if "POSE_DIR" in row and pd.notna(row["POSE_DIR"]) else ""
        pix = load_clip_frames(video, self.img_size, self.frames, self.fps, self.frame_stride, frame_drop_rate=self.frame_drop_rate)  # Pass frame_drop_rate
        pose = load_pose_or_none(pose_dir) # Load pose data
        return {"pixel_values": pix, "text": text, "pose": pose, "name": row.get("SENTENCE_NAME","")} # Return pose
