import torch as t
import torch.nn as nn
import numpy as np
from sklearn.metrics import label_ranking_loss


class RankingLoss:
    def __init__(self):
        self.name = "RankingLoss"

    def __call__(self, predict, target):
        predict = predict.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        predict[predict > 0.5] = 1
        predict[predict < 0.5] = 0
        p_shape = predict.shape
        t_shape = target.shape
        assert (p_shape[0] == t_shape[0])
        all_loss = 0.0
        for i in range(p_shape[0]):
            # predict[]
            all_loss = all_loss + label_ranking_loss(predict[i, :], target[i, :])
        return all_loss / p_shape[0]
