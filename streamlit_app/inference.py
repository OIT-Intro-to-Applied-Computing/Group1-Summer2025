import torch, os
from signlink.tokenization.spm import SentencePieceTokenizer
from signlink.models.sign2text_model import Sign2TextModel
from signlink.utils.video_io import load_clip_frames
from signlink.utils.pose_io import load_pose_or_none # New import

def load_model(ckpt_path, spm_path, d_model=512, dec_layers=4, nhead=8, use_pose=True, device="cpu"): # Changed default to True
    tok = SentencePieceTokenizer(spm_path)
    vocab_size = tok.sp.get_piece_size()
    model = Sign2TextModel(vocab_size=vocab_size, d_model=d_model, decoder_layers=dec_layers, nhead=nhead, use_pose=use_pose)
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd, strict=False); model.eval() # Allow partial loading
    return model, tok

# Add pose_dir argument to predict
def predict(video_path, pose_dir, model, tok, device="cpu", img_size=192, frames=32):
    try:
        pix = load_clip_frames(video_path, img_size=img_size, frames=frames)
        pix = pix.unsqueeze(0).to(device)
        
        pose = None
        if model.use_pose: # Only try to load pose if the model expects it
            pose = load_pose_or_none(pose_dir) # Load pose data
            if pose is None:
                print(f"Warning: Pose data not found for {video_path} at {pose_dir}. Model will run without pose data if possible.")
            else:
                pose = pose.unsqueeze(0).to(device) # Add batch dimension and move to device
        
        print(f"Debug: pix shape: {pix.shape}, dtype: {pix.dtype}")
        if pose is not None:
            print(f"Debug: pose shape: {pose.shape}, dtype: {pose.dtype}")
        else:
            print("Debug: pose is None")

        ids = model.generate(pix, pose=pose, max_len=64) # Pass pose to model
        print(f"Debug: Raw generated IDs: {ids}")
        prediction = tok.decode(ids)
        
        if not prediction:
            print(f"Warning: Model generated an empty prediction for {video_path}.")
            return "No prediction generated (empty output)."
            
        return prediction
    except Exception as e:
        print(f"Error during prediction for {video_path}: {e}")
        return f"Error during prediction: {e}"
