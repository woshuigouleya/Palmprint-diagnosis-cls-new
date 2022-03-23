import torch as t
import torch.nn.functional as F
import numpy as np
from backbone import SKNet, basic


class MyConv(t.nn.Module):
    def __init__(self, channels=3, kernel='filter1'):
        super(MyConv, self).__init__()
        self.channels = channels

        filter1 = [[-1, -1, -1],
                   [-1, 9, -1],
                   [-1, -1, -1]]

        filter2 = [[-1, -1, -1],
                   [0, 0, 0],
                   [1, 1, 1]]
        filter3 = [[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]]

        filter4 = [[-1, -1, 0],
                   [-1, 0, 1],
                   [0, 1, 1]]

        filter5 = [[0, 1, 1],
                   [-1, 0, 1],
                   [-1, -1, 0]]
        if kernel == 'filter1':
            kernel = filter1
        elif kernel == 'filter2':
            kernel = filter2
        elif kernel == 'filter3':
            kernel = filter3
        elif kernel == 'filter4':
            kernel = filter4
        elif kernel == 'filter5':
            kernel = filter5
        else:
            print('kernel error')
            exit(0)
        kernel = t.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = t.nn.Parameter(data=kernel, requires_grad=False)  # TODO: this line maybe error (kernel)

    def __call__(self, x):
        x = F.conv2d(x, self.weight, padding=1, groups=self.channels)
        return x


class MLLTnet(basic.BasicModule):
    def __init__(self, num, InputShape=(512, 512)):
        super(MLLTnet, self).__init__()
        self.numberClass = num
        self.enhancement = MyConv(kernel='filter1')
        self.prewitt1 = MyConv(kernel='filter2')
        self.prewitt2 = MyConv(kernel='filter3')
        self.prewitt3 = MyConv(kernel='filter4')
        self.prewitt4 = MyConv(kernel='filter5')

        # TODO: kernel_size & padding maybe wrong.
        self.input_rgb = t.nn.Sequential(
            t.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1),
            t.nn.BatchNorm2d(1),
            t.nn.ReLU(),
        )
        self.input_rgb_enhancement = t.nn.Sequential(
            t.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1),
            t.nn.BatchNorm2d(1),
            t.nn.ReLU(),
        )
        self.prewitt_4cat = t.nn.Sequential(
            t.nn.Conv2d(in_channels=12, out_channels=1, kernel_size=3, padding=1),
            t.nn.BatchNorm2d(1),
            t.nn.ReLU()
        )

        self.intput_to_sknet = t.nn.Sequential(
            t.nn.Conv2d(in_channels=5, out_channels=5, kernel_size=3, padding=1),
            t.nn.BatchNorm2d(5),
            t.nn.ReLU(),
            t.nn.Conv2d(in_channels=5, out_channels=3, kernel_size=3, padding=1),
            t.nn.BatchNorm2d(3),
            t.nn.ReLU()
        )
        self.SKNet = SKNet.SKNet(num_classes=self.numberClass, depth=50)
        self.outLayer1 = t.nn.Sequential(
            t.nn.Linear(1000, 512),
            t.nn.ReLU(),
            t.nn.Dropout(0.5))
        self.outLayer2 = t.nn.Linear(512, self.numberClass)

    def forward(self, rgb):
        # rgb_enhancement = self.enhancement(rgb)
        prewitt_1 = self.prewitt1(rgb)
        prewitt_2 = self.prewitt2(rgb)
        prewitt_3 = self.prewitt3(rgb)
        prewitt_4 = self.prewitt4(rgb)

        prewitt = t.cat([prewitt_1, prewitt_2, prewitt_3, prewitt_4], 1)
        x_rgb = self.input_rgb(rgb)
        # x_enhancement = self.input_rgb_enhancement(rgb_enhancement)
        x_prewitt_4cat = self.prewitt_4cat(prewitt)
        # x = t.cat([x_rgb, x_enhancement, x_prewitt_4cat, rgb], 1)
        x = t.cat([x_rgb, x_prewitt_4cat, rgb], 1)

        x = self.intput_to_sknet(x)
        x = self.SKNet(x)
        # x = self.outLayer1(x)
        # x = self.outLayer2(x)
        return x
