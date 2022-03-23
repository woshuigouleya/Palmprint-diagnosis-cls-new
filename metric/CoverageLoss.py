import torch as t
import torch.nn as nn
import numpy as np
from sklearn.metrics import coverage_error


class CoverageLoss:
    def __init__(self):
        self.name = "CoverageLoss"

    def __call__(self, predict, target):
        predict = predict.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        p_shape = predict.shape
        t_shape = target.shape
        assert (p_shape[0] == t_shape[0])
        all_loss = 0.0
        for i in range(p_shape[0]):
            # predict[]
            all_loss = all_loss + coverage_error(predict[i, :], target[i, :])
        return all_loss / p_shape[0]
