import torch
import torch.nn as nn
import torch.nn.functional as F

def beam_search(decoder_step, bos_id, eos_id, max_len=64, beam=4, device="cpu"):
    # decoder_step takes (prev_ids[B, L], enc_out, enc_pad_mask) -> logits[B, V]
    B = enc_batch = 1  # this simple version assumes batch size 1 during generation
    sequences = [(torch.tensor([bos_id], device=device), 0.0)]
    finished = []
    enc_out = None  # supplied via closure in caller, not used here directly
    for t in range(max_len):
        all_candidates = []
        for seq, score in sequences:
            if seq[-1].item() == eos_id:
                finished.append((seq, score))
                continue
            logits = decoder_step(seq.unsqueeze(0))  # [1, V]
            logp = F.log_softmax(logits[0], dim=-1)
            topk = torch.topk(logp, beam)
            for k in range(beam):
                next_id = topk.indices[k].unsqueeze(0)
                cand = (torch.cat([seq, next_id]), score + topk.values[k].item())
                all_candidates.append(cand)
        if not all_candidates and finished:
            break
        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam]
    if finished:
        best = max(finished, key=lambda x: x[1])[0]
    else:
        best = sequences[0][0]
    return best
