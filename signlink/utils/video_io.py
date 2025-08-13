import numpy as np, torch
from typing import Tuple
try:
    import decord
    from decord import VideoReader, cpu
    HAVE_DECORD = True
except Exception:
    import cv2
    HAVE_DECORD = False

def _resize_center_crop(frames, size=192):
    # frames: (T,H,W,3) uint8
    T,H,W,_ = frames.shape
    if HAVE_DECORD:
        import cv2
    import cv2 as cv
    out = []
    for t in range(T):
        img = frames[t]
        h, w = img.shape[:2]
        scale = size / min(h, w)
        nh, nw = int(round(h*scale)), int(round(w*scale))
        img = cv.resize(img, (nw, nh), interpolation=cv.INTER_AREA)
        # center crop
        y0 = (nh - size)//2; x0 = (nw - size)//2
        img = img[y0:y0+size, x0:x0+size]
        out.append(img)
    return np.stack(out, axis=0)

def load_clip_frames(path: str, img_size: int=192, frames: int=32, fps: int=25, frame_stride:int=2, frame_drop_rate: float=0.0): # Added frame_drop_rate
    # Returns tensor (C,T,H,W) float32 in [0,1]
    if HAVE_DECORD:
        vr = VideoReader(path, ctx=cpu(0))
        # sample indices
        total = len(vr)
        idxs = np.linspace(0, total-1, num=frames*frame_stride, dtype=int)[::frame_stride]
        
        # Apply temporal frame drop
        if frame_drop_rate > 0.0:
            num_to_drop = int(len(idxs) * frame_drop_rate)
            if num_to_drop > 0:
                drop_indices = np.random.choice(len(idxs), num_to_drop, replace=False)
                idxs = np.delete(idxs, drop_indices)
                if len(idxs) == 0: # Ensure at least one frame remains
                    idxs = np.array([np.random.randint(0, total-1)])
        
        batch = vr.get_batch(idxs).asnumpy()  # (T,H,W,3)
    else:
        import cv2
        cap = cv2.VideoCapture(path)
        imgs = []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(0, max(total-1,1), num=frames*frame_stride, dtype=int)[::frame_stride]

        # Apply temporal frame drop for OpenCV path too
        if frame_drop_rate > 0.0:
            num_to_drop = int(len(idxs) * frame_drop_rate)
            if num_to_drop > 0:
                drop_indices = np.random.choice(len(idxs), num_to_drop, replace=False)
                idxs = np.delete(idxs, drop_indices)
                if len(idxs) == 0: # Ensure at least one frame remains
                    idxs = np.array([np.random.randint(0, total-1)])

        want = set(idxs.tolist()); i=0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if i in want:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                imgs.append(frame)
            i+=1
        cap.release()
        if len(imgs)==0: raise RuntimeError(f"Could not read frames from {path}")
        batch = np.stack(imgs, axis=0)
    batch = _resize_center_crop(batch, size=img_size)
    batch = batch.astype("float32")/255.0
    batch = np.transpose(batch, (3,0,1,2))  # (C,T,H,W)
    return torch.from_numpy(batch)
