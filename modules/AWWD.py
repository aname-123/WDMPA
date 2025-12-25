import torch
from torch import nn
from torch.nn import functional as F


class HaarWavelet(nn.Module):
    def __init__(self, in_channels, grad=True):
        super(HaarWavelet, self).__init__()
        self.in_channels = in_channels

        self.haar_weights = torch.ones(4, 1, 2, 2)
        # h_horizontal    [[ 1, -1], [ 1, -1]]
        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1
        # v_vertical     [[ 1,  1], [-1, -1]]
        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1
        # d_diagonal     [[ 1, -1], [-1,  1]]
        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.in_channels, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = grad

    def forward(self, x, rev=False):
        if not rev:
            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.in_channels) / 4.0
            out = out.reshape([x.shape[0], self.in_channels, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], 4 * self.in_channels, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            out = x.reshape([x.shape[0], 4, self.in_channels, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.in_channels * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups=self.in_channels)


class AWWD(nn.Module):
    def __init__(self, dim_in, dim, need=False, fusion_type='weighted', use_gate=True):
        super(AWWD, self).__init__()
        self.need = need
        self.use_gate = use_gate
        self.fusion_type = fusion_type

        if need:
            self.first_conv = nn.Conv2d(dim_in, dim, kernel_size=1, padding=0)
            self.HaarWavelet = HaarWavelet(dim, grad=True)
            self.dim = dim
        else:
            self.HaarWavelet = HaarWavelet(dim_in, grad=True)
            self.dim = dim_in

        if fusion_type == 'sum':
            self.high_fusion = lambda h, v, d: h + v + d
        elif fusion_type == 'weighted':
            self.weight_h = nn.Parameter(torch.ones(1))
            self.weight_v = nn.Parameter(torch.ones(1))
            self.weight_d = nn.Parameter(torch.ones(1))
            self.high_fusion = lambda h, v, d: h * self.weight_h + v * self.weight_v + d * self.weight_d
        elif fusion_type == 'conv':
            self.high_fusion = nn.Conv2d(self.dim * 3, self.dim, kernel_size=1)

        if use_gate:
            self.gate_conv = nn.Sequential(
                nn.Conv2d(self.dim, self.dim, kernel_size=1),
                nn.BatchNorm2d(self.dim),
                nn.Sigmoid()
            )

        self.conv_after_wave = nn.Sequential(
            nn.Conv2d(self.dim * 2 if use_gate else self.dim*2, dim, kernel_size=1, padding=0),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.need:
            x = self.first_conv(x)
        haar = self.HaarWavelet(x, rev=False)
        a = haar.narrow(1, 0, self.dim)
        h = haar.narrow(1, self.dim, self.dim)
        v = haar.narrow(1, self.dim * 2, self.dim)
        d = haar.narrow(1, self.dim * 3, self.dim)

        if self.fusion_type == 'conv':
            high = self.high_fusion(torch.cat([h, v, d], dim=1))
        else:
            high = self.high_fusion(h, v, d)

        if self.use_gate:
            gate = self.gate_conv(a)
            gated_high = high * gate
            x = torch.cat([a, gated_high], dim=1)
        else:
            x = torch.cat([a, high], dim=1)

        x = self.conv_after_wave(x)
        return x

if __name__ == '__main__':
    dim_in = 3
    dim = 24
    input = torch.rand(1, dim_in, 224, 224)
    block = AWWD(dim_in=dim_in, dim=dim, need=False, fusion_type='weighted', use_gate=False)
    output = block(input)

    print("input size:", input.size())
    print("output size:", output.size())
