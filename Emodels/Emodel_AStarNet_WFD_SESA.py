import torch
from torch import nn
from modules.StarConv import ConvBN
from timm.layers import DropPath, trunc_normal_
from modules.AWWD import AWWD
from modules.AStarConv_MPA import Adaptive_Star_Block


class Emodel_StarNet(nn.Module):
    def __init__(self, base_dim=24, depths=[2, 2, 8, 3], mlp_ratio=4, drop_path_rate=0.0, **kwargs):
        super().__init__()
        self.in_channel = 24
        self.stem = nn.Sequential(ConvBN(3, self.in_channel, k=3, s=2, p=1), nn.ReLU6())
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            down_sampler = AWWD(self.in_channel, embed_dim, need=False, fusion_type='weighted', use_gate=False)
            self.in_channel = embed_dim
            blocks = [Adaptive_Star_Block(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))

        self.norm = nn.BatchNorm2d(self.in_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(9408, 2)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        bs, _, ny, nx = x.shape
        x = x.view(bs, -1)
        x = self.fc1(x)

        x = x.view(bs, 1, 2, 1, 1).permute(0, 1, 3, 4, 2).contiguous()
        return x

if __name__ == "__main__":
    model = Emodel_StarNet()
    image = torch.randn(1, 3, 224, 224)
    pred = model(image)
    print(pred.shape)
    print("finish!")

