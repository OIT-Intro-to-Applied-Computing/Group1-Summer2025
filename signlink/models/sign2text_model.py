import torch, torch.nn as nn, torch.nn.functional as F
from .encoders.frame_cnn import FrameCNN
from .encoders.temporal_cnn import Temporal1DEncoder
from .decoders.text_decoder_tfm import TextDecoderTFM, shift_for_teacher_forcing # Import helper

class Sign2TextModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, decoder_layers=4, nhead=8, use_pose=True, enc_layers=2):
        super().__init__()
        self.frame = FrameCNN(out_dim=d_model, trainable=False)
        self.temporal = Temporal1DEncoder(d_model=d_model, enc_layers=enc_layers, nhead=nhead)
        self.use_pose = use_pose
        if use_pose:
            self.pose_proj = nn.Sequential(nn.Linear(2*25, d_model//2), nn.ReLU(), nn.Dropout(0.1))
            self.fuse = nn.Linear(d_model + d_model//2, d_model)
        self.decoder = TextDecoderTFM(vocab_size=vocab_size, d_model=d_model, nhead=nhead, num_layers=decoder_layers)
        self.pad_id = 0; self.bos_id=1; self.eos_id=2
        self.label_smoothing = 0.1

    def encode(self, pixel_values, pose=None):
        # pixel_values: (B,C,T,H,W); pose: (B,T,J,2) or None
        rgb_seq = self.frame(pixel_values)           # (B,T,D)
        enc = self.temporal(rgb_seq)                 # (B,T',D)
        if self.use_pose and pose is not None:
            B,T,J,_ = pose.shape
            # pad/trim pose to match T' roughly by simple sub-sampling
            idx = torch.linspace(0, T-1, steps=enc.size(1)).long().to(pose.device)
            pose_t = pose[:, idx]                    # (B,T',J,2)
            pose_flat = pose_t.flatten(2)            # (B,T',J*2)
            pose_feat = self.pose_proj(pose_flat)    # (B,T',D/2)
            enc = self.fuse(torch.cat([enc, pose_feat], dim=-1))
        return enc

    def forward(self, pixel_values, tgt_ids=None, pose=None): # Added default None for tgt_ids
        enc = self.encode(pixel_values, pose)            # (B,Te,D)
        
        # Teacher-forcing shift: input to decoder is shifted right
        # e.g., if tgt_ids = [BOS, A, B, C, EOS, PAD],
        # decoder_input = [BOS, A, B, C, EOS]
        # targets = [A, B, C, EOS, PAD]
        tgt_inp, targets, tgt_out_mask = shift_for_teacher_forcing(tgt_ids, pad_id=self.pad_id) # Use helper

        logits = self.decoder(enc, tgt_inp)              # (B,L-1,V)
        
        # label smoothed CE
        logp = torch.log_softmax(logits, dim=-1)
        B,L_minus_1,V = logp.shape # L-1 because decoder_input is shifted
        with torch.no_grad():
            true = torch.zeros_like(logp).scatter_(-1, targets.unsqueeze(-1), 1.0) # Use shifted targets
            smooth = self.label_smoothing / (V - 1)
            true = true * (1 - self.label_smoothing) + (1 - true) * smooth
        loss = -(true * logp).sum(-1)
        pad_mask = (targets == self.pad_id) # Use shifted targets for pad_mask
        loss = (loss.masked_fill(pad_mask, 0).sum() / (~pad_mask).sum().clamp_min(1))
        return loss

    @torch.no_grad()
    def generate(self, pixel_values=None, pose=None, max_len=64, min_len=5, length_penalty=0.6): # Added min_len, length_penalty
        device = (pixel_values.device if pixel_values is not None else pose.device)
        enc = self.encode(pixel_values, pose)      # (B,Te,D)
        # batch size 1 decoding for simplicity
        B = enc.size(0)
        assert B==1, "Generation assumes batch=1 for MVP"
        ids = torch.tensor([[self.bos_id]], device=device, dtype=torch.long)
        for t in range(max_len): # Changed loop variable to t
            logits = self.decoder(enc, ids)[:, -1, :]  # (1,V)
            if t < min_len: # Forbid EOS early
                logits[0, self.eos_id] = -1e9
            # nudge against early EOS
            logits[0, self.eos_id] -= length_penalty / (t + 1)
            next_id = torch.argmax(logits, dim=-1, keepdim=True)  # greedy for MVP
            ids = torch.cat([ids, next_id], dim=1)
            if next_id.item() == self.eos_id and t >= min_len: break # Check min_len before breaking
        return ids[0].tolist()
