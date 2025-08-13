import torch, pandas as pd
from torch.utils.data import Dataset
from ..utils.pose_io import load_pose_or_none

class How2SignPose(Dataset):
    def __init__(self, tsv_path):
        self.df = pd.read_csv(tsv_path, sep="\t")
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pose_dir = str(row["POSE_DIR"]) if "POSE_DIR" in row and pd.notna(row["POSE_DIR"]) else ""
        pose = load_pose_or_none(pose_dir)
        return {"pose": pose}
