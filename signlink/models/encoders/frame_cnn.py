import torch
import torch.nn as nn
import torchvision.models as models

class FrameCNN(nn.Module):
    def __init__(self, name="resnet18", out_dim=512, trainable=False):
        super().__init__()
        if name == "resnet18":
            m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            feat_dim = 512
            layers = list(m.children())[:-2]  # keep conv5_x, spatial map
            self.backbone = nn.Sequential(*layers)
            self.pool = nn.AdaptiveAvgPool2d((1,1))
        else:
            raise ValueError("Unsupported backbone")
        self.proj = nn.Linear(feat_dim, out_dim)
        if not trainable:
            for p in self.backbone.parameters(): p.requires_grad = False

    def forward(self, frames):  # (B,C,T,H,W)
        B,C,T,H,W = frames.shape
        x = frames.permute(0,2,1,3,4).reshape(B*T, C, H, W)  # (B*T,C,H,W)
        feat = self.backbone(x)                               # (B*T, C', h, w)
        feat = self.pool(feat).flatten(1)                     # (B*T, C')
        feat = self.proj(feat)                                # (B*T, D)
        feat = feat.view(B, T, -1)                            # (B,T,D)
        return feat
