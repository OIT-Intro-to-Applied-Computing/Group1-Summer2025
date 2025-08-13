import torch, torch.nn as nn

class Temporal1DEncoder(nn.Module):
    def __init__(self, d_model=512, channels=[512,512,512,512], kernels=[5,5,3,3], strides=[2,1,2,1], dropout=0.1, enc_layers=2, nhead=8):
        super().__init__()
        assert len(channels)==len(kernels)==len(strides)
        layers = []
        in_ch = d_model
        for ch, k, s in zip(channels, kernels, strides):
            layers += [nn.Conv1d(in_ch, ch, k, stride=s, padding=k//2), nn.ReLU(), nn.Dropout(dropout)]
            in_ch = ch
        self.conv = nn.Sequential(*layers)
        self.proj = nn.Linear(in_ch, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=enc_layers) if enc_layers>0 else None

    def forward(self, seq):  # (B,T,D)
        x = seq.permute(0,2,1)         # (B,D,T)
        x = self.conv(x)               # (B,C,T')
        x = x.permute(0,2,1)           # (B,T',C)
        x = self.proj(x)               # (B,T',D)
        if self.enc is not None:
            x = self.enc(x)            # (B,T',D)
        return x
