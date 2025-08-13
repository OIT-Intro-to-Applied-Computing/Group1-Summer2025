import os, json, numpy as np, torch

def load_pose_or_none(pose_dir):
    if not pose_dir or not os.path.isdir(pose_dir):
        return None
    # try to read *.json ordered by filename
    files = sorted([f for f in os.listdir(pose_dir) if f.endswith('.json')])
    if not files:
        return None
    seq = []
    for f in files:
        with open(os.path.join(pose_dir, f), 'r') as fp:
            data = json.load(fp)
        # assume 'people'[0]['pose_keypoints_2d']-like or generic list of (x,y[,c])
        arr = None
        if isinstance(data, dict) and "people" in data and data["people"]:
            pts = data["people"][0].get("pose_keypoints_2d", [])
            arr = np.array(pts, dtype=np.float32).reshape(-1,3) if pts else None
        elif isinstance(data, list):
            arr = np.array(data, dtype=np.float32)
        if arr is None:
            continue
        seq.append(arr[:, :2])  # (J,2)
    if not seq: return None
    seq = np.stack(seq, axis=0)  # (T,J,2)
    # simple normalization: subtract neck (joint 1 if exists), scale by torso
    J = seq.shape[1]
    neck = seq[:, min(1,J-1), :]
    seq = seq - neck[:,None,:]
    scale = (np.linalg.norm(seq[:,0,:]-seq[:,min(8,J-1),:], axis=-1, keepdims=True)+1e-6)  # shoulder-hip approx
    seq = seq / scale[:,None,:]
    
    # Add pose jitter (small random noise) for augmentation
    # Gated by a flag or env var if needed, for now, always apply
    seq = seq + 0.01 * np.random.randn(*seq.shape).astype(np.float32) # Add jitter

    return torch.from_numpy(seq.astype(np.float32))  # (T,J,2)
