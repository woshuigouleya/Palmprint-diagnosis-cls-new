import os
import sys

sys.path.append(os.path.split(sys.path[0])[0])
import torch as t
from backbone import basic, SKNet
from backbone.PrewittConv import PrewittConv


class FusePrewittBranch(basic.BasicModule):
    def __init__(self, num, InputShape=(512, 512)):
        super(FusePrewittBranch, self).__init__()
        self.numberClass = num
        self.prewitt1 = PrewittConv(kernel='filter2')
        self.prewitt2 = PrewittConv(kernel='filter3')
        self.prewitt3 = PrewittConv(kernel='filter4')
        self.prewitt4 = PrewittConv(kernel='filter5')
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
            t.nn.Conv2d(in_channels=12, out_channels=4, kernel_size=3, padding=1),
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

    def forward(self, rgb):
        prewitt_1 = self.prewitt1(rgb)
        prewitt_2 = self.prewitt2(rgb)
        prewitt_3 = self.prewitt3(rgb)
        prewitt_4 = self.prewitt4(rgb)
        prewitt_fe = t.cat([prewitt_1, prewitt_2, prewitt_3, prewitt_4], 1)
        Prewitt = self.FE_Other(prewitt_fe)
        Ori = self.FE_Ori(rgb)
        fuse = self.SKNet(Ori, Prewitt)
        return fuse
