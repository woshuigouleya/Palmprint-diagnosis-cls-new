import os
import sys

sys.path.append(os.path.split(sys.path[0])[0])

import torch as t
import torch.nn.functional as F
import numpy as np

from backbone import basic, SKNet
from opZoo.Gabor_op import GaborMethod


class GaborSingleBranch(basic.BasicModule):
    def __init__(self, num, InputShape=(512, 512)):
        super(GaborSingleBranch, self).__init__()
        self.numberClass = num
        self.gap = t.nn.AdaptiveAvgPool2d(1)
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
        self.CLS = t.nn.Sequential(
            t.nn.Linear(32, self.numberClass)
        )

    def forward(self, rgb):
        x = self.FE(rgb)
        x = self.SKNet(x)
        return x
