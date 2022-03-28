import os
import sys

sys.path.append(os.path.split(sys.path[0])[0])
import torch as t
from backbone import basic, SKNet
from backbone.PrewittConv import PrewittConv


class FuseGaborBranch(basic.BasicModule):
    def __init__(self, num, InputShape=(512, 512)):
        super(FuseGaborBranch, self).__init__()
        self.numberClass = num
        self.FE_Ori = t.nn.Sequential(
            t.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, padding=1),
            t.nn.BatchNorm2d(4),
            t.nn.ReLU(),
            t.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1),
            t.nn.BatchNorm2d(4),
            t.nn.ReLU(),
            t.nn.Conv2d(in_channels=4, out_channels=3, kernel_size=3, padding=1),
            t.nn.BatchNorm2d(3),
            t.nn.ReLU()
        )
        self.FE_Other = t.nn.Sequential(
            t.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, padding=1),
            t.nn.BatchNorm2d(4),
            t.nn.ReLU(),
            t.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1),
            t.nn.BatchNorm2d(4),
            t.nn.ReLU(),
            t.nn.Conv2d(in_channels=4, out_channels=3, kernel_size=3, padding=1),
            t.nn.BatchNorm2d(3),
            t.nn.ReLU()
        )

        for p in self.parameters():
            p.requires_grad = False

        self.SKNet = SKNet.fuse_sknet(num_classes=self.numberClass)

    def forward(self, rgb,Gabor):
        Other = self.FE_Other(Gabor)
        Ori = self.FE_Ori(rgb)
        fuse = self.SKNet(Ori, Other)
        return fuse
