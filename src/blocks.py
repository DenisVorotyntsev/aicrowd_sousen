import torch


class SpatialDropout1d(torch.nn.Module):
    def __init__(self, p):
        super(SpatialDropout1d, self).__init__()
        self.dropout = torch.nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # bs, n_ch, len -> bs, len, n_ch
        x = x.permute(0, 2, 1)

        # bs, len, n_ch -> bs, len, n_ch, 1 (droput2d: N,[C],H,W)
        x = x.unsqueeze(3)
        x = self.dropout(x)
        x = x.squeeze(3)

        # bs, len, n_ch -> bs, n_ch, len
        x = x.permute(0, 2, 1)
        return x


class ResnetBlock(torch.nn.Module):
    # idea is taken from http://torch.ch/blog/2016/02/04/resnets.html
    # and https://arxiv.org/pdf/1812.01187v2.pdf (resnet D)
    def __init__(self, in_channels: int = 64, out_channels: int = 128):
        super().__init__()
        # left
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn1 = torch.nn.BatchNorm1d(out_channels)
        self.relu1 = torch.nn.ReLU(out_channels)

        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn2 = torch.nn.BatchNorm1d(out_channels)
        self.relu2 = torch.nn.ReLU(out_channels)

        self.conv3 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        # right
        self.avg_pool = torch.nn.AvgPool1d(kernel_size=2, stride=2)
        self.conv_right = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # left
        left = self.conv1(x)
        left = self.bn1(left)
        left = self.relu1(left)

        left = self.conv2(left)
        left = self.bn2(left)
        left = self.relu2(left)

        left = self.conv3(left)
        left = self.bn3(left)

        # right
        right = self.avg_pool(x)
        right = self.conv_right(right)

        out = left + right
        return out


def cnn_block(
    in_channels,
    out_channels,
    kernel_size=3,
    pooling_ks=3,
):
    cnn_block_ = torch.nn.Sequential(
        torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
        ),
        torch.nn.BatchNorm1d(out_channels),
        torch.nn.ReLU(),

        torch.nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
        ),
        torch.nn.BatchNorm1d(out_channels),
        torch.nn.ReLU(),

        torch.nn.MaxPool1d(kernel_size=pooling_ks),
    )
    return cnn_block_

