import math

import torch.nn as nn
import torch


class Supervised1dModel(nn.Module):
    def __init__(self, backbone, backbone_output_dim: int, num_classes=1):
        super().__init__()
        self.backbone = backbone

        self.linear_block = nn.Sequential(
            nn.Linear(backbone_output_dim, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, num_classes),
        )

    def forward(self, inp):
        x = self.backbone(inp)
        out = self.linear_block(x)
        return out
