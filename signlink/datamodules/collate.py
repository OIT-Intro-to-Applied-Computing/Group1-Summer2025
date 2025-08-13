import torch

def collate_fn(batch, tokenizer):
    # batch: list of dicts with pixel_values, text, pose_dir, name
    B = len(batch)
    # video
    maxT = max(x["pixel_values"].shape[1] for x in batch)
    C,H,W = batch[0]["pixel_values"].shape[0], batch[0]["pixel_values"].shape[2], batch[0]["pixel_values"].shape[3]
    vids = torch.zeros(B, C, maxT, H, W, dtype=torch.float32)
    for i, x in enumerate(batch):
        T = x["pixel_values"].shape[1]
        vids[i, :, :T] = x["pixel_values"]
    # text ids
    ids = [torch.tensor(tokenizer.encode(x["text"]), dtype=torch.long) for x in batch]
    L = max(len(t) for t in ids)
    tgt = torch.zeros(B, L, dtype=torch.long)
    tgt_mask = torch.ones(B, L, dtype=torch.bool)
    for i, t in enumerate(ids):
        tgt[i, :len(t)] = t
        tgt_mask[i, :len(t)] = False
    # we don't load pose here; inference helper will handle pose if available
    # pose
    # Check if pose data is present in the batch and collate it
    if "pose" in batch[0] and batch[0]["pose"] is not None:
        # Filter out None poses and get max sequence length
        valid_poses = [x["pose"] for x in batch if x["pose"] is not None]
        if valid_poses:
            max_pose_T = max(p.shape[0] for p in valid_poses)
            num_keypoints = valid_poses[0].shape[1] # N
            num_coordinates = valid_poses[0].shape[2] # C
            # Initialize poses with all three dimensions: (B, max_T, N, C)
            poses = torch.zeros(B, max_pose_T, num_keypoints, num_coordinates, dtype=torch.float32)
            for i, x in enumerate(batch):
                if x["pose"] is not None:
                    T_pose = x["pose"].shape[0]
                    poses[i, :T_pose] = x["pose"] # This assignment should now work
                # If pose is None, the corresponding row in 'poses' will remain zeros
        else:
            poses = None # No valid poses in batch
    else:
        poses = None # No pose key in batch items

    names = [x.get("name","") for x in batch]
    # pose_dirs = [x.get("pose_dir","") for x in batch] # pose_dir is no longer needed here
    return {"pixel_values": vids, "tgt_ids": tgt, "tgt_mask": tgt_mask, "names": names, "pose": poses} # Return pose
