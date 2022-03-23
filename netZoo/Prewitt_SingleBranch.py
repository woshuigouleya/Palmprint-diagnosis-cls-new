import os
import sys

sys.path.append(os.path.split(sys.path[0])[0])
import torch as t
from backbone import basic, SKNet
from backbone.PrewittConv import PrewittConv


class PrewittSingleBranch(basic.BasicModule):
    def __init__(self, num, InputShape=(512, 512)):
        super(PrewittSingleBranch, self).__init__()
        self.numberClass = num
        self.prewitt1 = PrewittConv(kernel='filter2')
        self.prewitt2 = PrewittConv(kernel='filter3')
        self.prewitt3 = PrewittConv(kernel='filter4')
        self.prewitt4 = PrewittConv(kernel='filter5')
        self.FE = t.nn.Sequential(
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
        self.SKNet = SKNet.SKNet(num_classes=self.numberClass, depth=50)
        self.CLS = t.nn.Sequential(
            t.nn.Linear(32, self.numberClass)
        )

    def forward(self, rgb):
        prewitt_1 = self.prewitt1(rgb)
        prewitt_2 = self.prewitt2(rgb)
        prewitt_3 = self.prewitt3(rgb)
        prewitt_4 = self.prewitt4(rgb)
        prewitt_fe = t.cat([prewitt_1, prewitt_2, prewitt_3, prewitt_4], 1)
        x = self.FE(prewitt_fe)
        x = self.SKNet(x)
        return x
