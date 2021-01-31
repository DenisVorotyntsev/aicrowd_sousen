import numpy as np
import torch.nn as nn
import torch

from blocks import cnn_block, ResnetBlock


class VGGNet(nn.Module):
    def __init__(self, in_channels: int = 128, n_filters: int = 256):
        super().__init__()
        self.cnn_block1 = cnn_block(in_channels=in_channels, out_channels=n_filters, kernel_size=7)
        self.cnn_block2 = cnn_block(in_channels=n_filters, out_channels=n_filters*2, kernel_size=5)
        self.cnn_block3 = cnn_block(in_channels=n_filters*2, out_channels=n_filters*2, kernel_size=3)

    def forward(self, x):
        # x input shape: batch_size, sequence_length, n_features
        # cnn input/output: batch_size, n_features, sequence_length
        x = x.permute(0, 2, 1)
        x1 = self.cnn_block1(x)
        x2 = self.cnn_block2(x1)
        x3 = self.cnn_block3(x2)

        concat = torch.cat([
            torch.nn.functional.adaptive_avg_pool1d(x3, 1),
            torch.nn.functional.adaptive_max_pool1d(x3, 1),
        ], dim=1)

        out = torch.squeeze(concat, dim=2)
        return out


class ResNetCombinedLayersFeatures(nn.Module):
    def __init__(self, in_channels: int = 128, n_filters: int = 256):
        super().__init__()
        self.cnn_block1 = ResnetBlock(in_channels, n_filters)
        self.cnn_block2 = ResnetBlock(n_filters, n_filters*2)
        self.cnn_block3 = ResnetBlock(n_filters*2, n_filters*2)

    def forward(self, x):
        # x input shape: batch_size, sequence_length, n_features
        # cnn input/output: batch_size, n_features, sequence_length
        x = x.permute(0, 2, 1)
        x1 = self.cnn_block1(x)
        x2 = self.cnn_block2(x1)
        x3 = self.cnn_block3(x2)

        concat = torch.cat([
            torch.nn.functional.adaptive_avg_pool1d(x2, 1),
            torch.nn.functional.adaptive_avg_pool1d(x3, 1),

            torch.nn.functional.adaptive_max_pool1d(x2, 1),
            torch.nn.functional.adaptive_max_pool1d(x3, 1),
        ], dim=1)

        out = torch.squeeze(concat, dim=2)
        return out


if __name__ == "__main__":
    model = VGGNet(in_channels=128)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("num params: ", pytorch_total_params)

    x = np.random.rand(256, 128)[np.newaxis, ...]
    x = torch.Tensor(x)
    y = model(x)
    print(y.size())

