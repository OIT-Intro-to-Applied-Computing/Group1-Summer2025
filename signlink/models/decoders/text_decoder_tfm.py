import torch, torch.nn as nn, torch.nn.functional as F

# At top-level in this file
def shift_for_teacher_forcing(tgt_ids, pad_id=0):
    # tgt_inp: BOS..last-1,  tgt_out: next tokens (incl EOS)
    tgt_inp = tgt_ids[:, :-1].contiguous()
    tgt_out = tgt_ids[:, 1:].contiguous()
    tgt_out_mask = (tgt_out == pad_id)
    return tgt_inp, tgt_out, tgt_out_mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1,L,D)
    def forward(self, x, start=0):
        return x + self.pe[:, start:start+x.size(1)]

class TextDecoderTFM(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=4, dropout=0.1, pad_id=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model)
        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.dec = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, vocab_size)
        self.pad_id = pad_id

    def forward(self, enc_out, tgt_inp): # Changed tgt_ids to tgt_inp
        # enc_out: (B,Te,D), tgt_inp: (B,L) - now shifted input
        tgt = self.embed(tgt_inp)
        tgt = self.pos(tgt)
        # autoregressive mask
        L = tgt.size(1)
        causal = torch.triu(torch.ones(L, L, device=tgt.device), diagonal=1).bool()
        tgt_pad = (tgt_inp == self.pad_id) # Use tgt_inp for padding mask
        out = self.dec(tgt, enc_out, tgt_mask=causal, tgt_key_padding_mask=tgt_pad)
        logits = self.out(out)
        return logits
