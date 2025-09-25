import torch
import torch.nn as nn
import torch
import torch.nn as nn

class MPA(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(MPA, self).__init__()

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        self.sigmoid_channel = nn.Sigmoid()
        self.conv3x3 = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, dilation=1, bias=False)
        self.conv5x5 = nn.Conv2d(in_channels, 1, kernel_size=3, padding=2, dilation=2, bias=False)
        self.conv7x7 = nn.Conv2d(in_channels, 1, kernel_size=3, padding=3, dilation=3, bias=False)

        self.conv_fuse = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):

        b, c, _, _ = x.size()
        y = self.global_avg_pool(x)
        y = self.conv1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y_channel = self.sigmoid_channel(y)
        x_channel = x * y_channel

        out_3x3 = self.conv3x3(x_channel)
        out_5x5 = self.conv5x5(x_channel)
        out_7x7 = self.conv7x7(x_channel)

        out_fused = torch.cat([out_3x3, out_5x5, out_7x7], dim=1)
        out_fused = self.conv_fuse(out_fused)

        y_spatial = self.sigmoid_spatial(out_fused)
        return x_channel * y_spatial

