import torch as t
import torch.nn.functional as F
import numpy as np


class PrewittConv(t.nn.Module):
    def __init__(self, channels=3, kernel='filter1'):
        super(PrewittConv, self).__init__()
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
        self.weight = t.nn.Parameter(data=kernel, requires_grad=False).cuda()  # TODO: this line maybe error (kernel)

    def __call__(self, x):
        x = F.conv2d(x, self.weight, padding=1, groups=self.channels)
        return x
