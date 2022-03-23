import os
import sys

sys.path.append(os.path.split(sys.path[0])[0])
import torch as t
from backbone import basic, SKNet

class OriginalSingleBranch(basic.BasicModule):
    def __init__(self, num, InputShape=(512, 512)):
        super(OriginalSingleBranch, self).__init__()
        self.numberClass = num
        self.FE = t.nn.Sequential(
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
        self.SKNet = SKNet.SKNet(num_classes=self.numberClass, depth=50)

    def forward(self, rgb):
        x = self.FE(rgb)
        x = self.SKNet(x)
        return x