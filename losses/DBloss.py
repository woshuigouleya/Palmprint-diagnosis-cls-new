import logging

import torch as t
import torch.nn.functional as F
from torch.autograd import Variable


class DBLoss(t.nn.Module):
    def __init__(self, num_cls, num, alpha, beta, miu, lambada, k):
        """

        :param num_cls:         44 class in this project
        :param num:             num of all labels.
        :param alpha:
        :param beta:
        :param miu:
        :param lambada:
        :param k:
        """
        super(DBLoss, self).__init__()
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")
        self.C = t.tensor(num_cls).float().to(self.device)
        self.num = t.tensor(num).float().to(self.device)
        self.alpha = t.tensor(alpha).float().to(self.device)
        self.beta = t.tensor(beta).float().to(self.device)
        self.miu = t.tensor(miu).float().to(self.device)
        self.lambada = t.tensor(lambada).float().to(self.device)
        self.k = t.tensor(k).float().to(self.device)
        self.class_freq = None
        self.neg_class_freq = None
        self.freq_inv = None

    def host_update(self, each_num_cls):
        """
        Before each epoch, U should use host_update function to update
        self.each_num_cls etc. TODO
        :param each_num_cls:
        :return:
        """
        neg_num_cls = [self.num - item for item in each_num_cls]
        self.class_freq = t.tensor(each_num_cls).float().to(self.device)
        self.neg_class_freq = t.tensor(neg_num_cls).float().to(self.device)
        self.freq_inv = t.ones(self.class_freq.shape).float().to(self.device) / self.class_freq  # 1 / n_{i}
        self.freq_inv = self.freq_inv.to(self.device)

    def re_balance_weight(self, gt_labels):
        r"""
        $$ \frac{P_{i}^{C}(x^{k})}{P^{I}(x^{k})} $$
        :param gt_labels:
        :return:
        """
        gt_labels = gt_labels.to(self.device)
        repeat_rate = t.sum(gt_labels.float() * self.freq_inv, dim=1, keepdim=True)
        pos_weight = self.freq_inv.clone().detach().unsqueeze(0) / repeat_rate
        # pos and neg are equally treated
        weight = t.sigmoid(self.beta * (pos_weight - self.miu)) + self.alpha
        return weight

    def prior_nu(self):
        p_i_inv = self.num / self.class_freq
        b_i_hat = -t.log(p_i_inv - 1)
        nu = -self.k * b_i_hat
        return nu

    def forward(self, predict, target):
        predict = predict.to(self.device)
        target = target.to(self.device)
        r = self.re_balance_weight(target)  # Vector, weight_r_{i}
        nu = self.prior_nu()
        loss_1 = target * t.log(t.tensor(1.0).float().to(self.device) + t.exp(nu - predict))
        lambada_inv = t.tensor(1).float().to(self.device) / self.lambada
        loss_2_1 = (t.ones(target.shape).float().to(self.device) - target)
        loss_2_2 = t.log(t.tensor(1).float().cuda() + t.exp(self.lambada * (predict - nu)))
        loss = loss_1 + lambada_inv * (loss_2_1 * loss_2_2)
        loss = r * loss
        loss = t.sum(loss, dim=1)
        loss = loss / self.C
        return loss.mean()


if __name__ == "__main__":
    logging.warning("For DB loss function debug only.")
    exit(1)
    batch_size = 16
    _num_cls = 4
    _num = 100
    _each_num_cls = [10, 20, 30, 40]
    _alpha = 0.1
    _beta = 10
    _miu = 0.3
    _lambda = 5
    _k = 0.05
    Loss = DBLoss(_num_cls, _num, _alpha, _beta, _miu, _lambda, _k)
    Loss.host_update(_each_num_cls)
    p = t.rand((batch_size, 4)).float().cuda()
    tar = t.rand((batch_size, 4)).float().cuda()
    # print(p, p.shape)
    x = Variable(p, requires_grad=True)
    y = Variable(tar, requires_grad=True)
    loss = Loss(x, y)
    loss.backward()
    print(loss)
