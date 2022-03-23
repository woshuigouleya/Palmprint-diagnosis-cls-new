import os
import sys
import logging

import cv2
from PIL import Image

import numpy as np
from tqdm import tqdm

import torch as t
from torch.autograd import Variable
from torchvision import transforms as T
import torch.nn.functional as F
from torch.utils import data

from warmup_scheduler import GradualWarmupScheduler

from losses import DBloss, FocalLoss, ResamplerLoss
from metric import HammingLoss, CoverageLoss, RankingLoss
from sampler import CWsampler
from utils import EMA
from utils.logger import FileLogger
from utils.WorkShop import WorkManager
from utils.memter import MemData
from utils.VisdomVisualizer import Visualizer
from opZoo.Gabor_op import GaborMethod

"""
1. MLLTnet for training, optimizer, scheduler.
2. Metric, figure.
3. dic, logger.
"""


def load_model(net, path, device, separate=False):
    if separate:
        net = net.to(device)
        checkpoints = t.load(path)
        new = {}
        for k, v in checkpoints.items():
            if "prewitt" not in k:
                new.update({k: v})
        net.load_state_dict(new)
    else:
        net.load(path)
        net.to(device)


class ClsData(data.Dataset):
    """
    NOTE: 1. The rate of \frac{num_train}{n_test} should be set in CWSampler
          2. You should rebuild this ClsData-Training each epoch. But Test don't need.
    """

    def __init__(self, rhs, trans=None, GaborFuse=False):
        self.rhs = rhs  # rhs is a data struct build in get_data.
        self._data = rhs.data()
        self.trans = trans
        self.GaborFuse = GaborFuse
        if self.trans is None:
            self.trans = T.Compose([
                T.ToTensor(),
                T.Normalize((0.57,), (0.18,))
            ])

    def __getitem__(self, item):
        img_path = self._data[item]['index']
        gt = self._data[item]['gt']
        img_path = img_path.replace("/media/liu/WangChengHua-Dis/cls-44/data/", "/home/liu/gyq/data/")
        ori_img = cv2.imread(img_path)
        if self.GaborFuse == False:
            # ori_img = GaborMethod.Enhance(ori_img)  # TODO need this line if Gabor required.
            ori_img = Image.fromarray(cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB))
            ori_img = self.trans(ori_img)
            target = t.FloatTensor(gt)
            return ori_img, target
        else:
            Gabor_img = GaborMethod.Enhance(ori_img)
            Gabor_img = Image.fromarray(cv2.cvtColor(Gabor_img, cv2.COLOR_BGR2RGB))
            Gabor_img = self.trans(Gabor_img)
            ori_img = Image.fromarray(cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB))
            ori_img = self.trans(ori_img)
            target = t.FloatTensor(gt)
            return ori_img, Gabor_img, target

    def __len__(self):
        return len(self._data)


class run:
    def __init__(self, networks, opt):
        self.opt = opt
        self.pbar = None
        self.networks = networks
        self.EMA_NET = None
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")
        self.optimizer = run.build_sgd_optimizer(self.networks, self.opt)
        self.__ecn = self.opt.each_cls_num
        # self.LOSS = run.build_resampler_loss_class(self.__ecn, [self.opt.total_num-item for item in self.__ecn])
        # self.optimizer = run.build_adam_optimizer(self.networks, self.opt)
        self.sampler = run.build_cw_sampler(self.opt)
        self.test_base_data = None
        self.train_base_data = None
        self.train_dataloader = None
        self.test_dataloader = None
        self.FOCALLOSS = FocalLoss.FocalLoss()
        # -- scheduler
        if self.opt.WarmUpEpoch != 0:
            self.scheduler = run.build_warmup_scheduler(self.optimizer, self.opt)
        else:
            self.scheduler = run.build_scheduler(self.optimizer, self.opt)
        # -- WorkShop
        self.WorkManager = WorkManager(self.opt.main_path)
        self.FileLogger = FileLogger(os.path.join(self.WorkManager.LogPath, "logger.txt"))
        # -- training memory of loss, acc, etc.
        self.MemLoss = MemData()
        self.MemHammingLoss = MemData()
        self.MemRankingLoss = MemData()
        self.MemCoverageLoss = MemData()
        # -- visdom visual
        self.visual = Visualizer("Running")

    @staticmethod
    def build_data_loader(sampler, opt, mode):
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.57,), (0.18,))
        ])
        train_base_data, test_base_data = run.build_base_data(sampler, opt, mode)
        train_loader = run.build_train_dataloader(train_base_data, transform, opt)
        test_loader = run.build_test_dataloader(test_base_data, transform, opt)
        return [train_base_data, test_base_data], [train_loader, test_loader]

    @staticmethod
    def build_base_data(sampler, opt, mode):
        if mode != "Full":
            sampler.step(opt.n_max_sampler)
        return sampler.data("Train"), sampler.data("Test")

    @staticmethod
    def build_test_dataloader(base_data, trans, opt):
        test_data = ClsData(base_data, trans)
        test_data_loader = data.DataLoader(test_data, batch_size=opt.batch_size_test,
                                           shuffle=opt.shuffle_test, num_workers=opt.num_workers_test)
        return test_data_loader

    @staticmethod
    def build_train_dataloader(base_data, trans, opt):
        train_data = ClsData(base_data, trans)
        train_data_loader = data.DataLoader(train_data, batch_size=opt.batch_size_train,
                                            shuffle=opt.shuffle_train, num_workers=opt.num_workers_train)
        return train_data_loader

    @staticmethod
    def build_sgd_optimizer(networks, opt):
        return t.optim.SGD([{'params': networks.parameters(), 'initial_lr': opt.lr}], lr=opt.lr, momentum=opt.momentum,
                           weight_decay=opt.weight_decay)

    @staticmethod
    def build_adam_optimizer(networks, opt):
        return t.optim.Adam(networks.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    @staticmethod
    def build_cw_sampler(opt):
        return CWsampler.CWSampler(opt.index_path, opt.NeedCls, opt.rate)

    @staticmethod
    def calculate_metric(predict, target):
        hamming_loss = HammingLoss.HammingLoss()(predict, target)
        # coverage_loss = CoverageLoss.CoverageLoss()(predict, target)
        # ranking_loss = RankingLoss.RankingLoss()(predict, target)
        return {"HammingLoss": hamming_loss}

    @staticmethod
    def build_scheduler(optimizer, opt):
        return t.optim.lr_scheduler.MultiStepLR(optimizer, opt.milestones, opt.scheduler_gamma, opt.last_epoch)

    @staticmethod
    def build_warmup_scheduler(optimizer, opt):
        ans = t.optim.lr_scheduler.MultiStepLR(optimizer, opt.milestones, opt.scheduler_gamma, opt.last_epoch)
        return GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=opt.WarmUpEpoch, after_scheduler=ans)

    def calculate_loss(self, predict, target, num_each_cls, num, num_cls):
        """

        :param predict:
        :param target:
        :param num_each_cls:
        :param num:             total training samples num.
        :param num_cls:         44 class
        :return:                loss in tensor.
        """
        # predict.to(self.device)
        # target.to(self.device)
        # target = target.long()
        db_loss = DBloss.DBLoss(num_cls=num_cls, num=num, alpha=self.opt.DB_alpha, beta=self.opt.DB_beta,
                                miu=self.opt.DB_miu, lambada=self.opt.DB_lambda, k=self.opt.DB_k)
        db_loss.host_update(each_num_cls=num_each_cls)
        focal_loss = FocalLoss.FocalLoss(gamma=self.opt.FO_gamma, alpha=self.opt.FO_alpha)
        return db_loss(predict, target) + focal_loss(predict, target)
        # return db_loss(predict, target)

    @staticmethod
    def build_resampler_loss_class(class_freq, neg_class_freq):
        return ResamplerLoss.ResampleLoss(class_freq, neg_class_freq)

    @staticmethod
    def calculate_bce_loss(predict, target):
        return t.nn.BCELoss(predict, target)

    def save_model(self, epoch):
        self.networks.save(path="need?",
                           name=os.path.join(self.WorkManager.ori_model_path, "Epoch" + str(epoch) + ".pth"))
        if self.opt.enable_EMA:
            self.EMA_NET.model.save(path="need?",
                                    name=os.path.join(self.WorkManager.ema_model_path, "Epoch" + str(epoch) + ".pth"))

    def print_model(self):
        # 打印模型的结构
        print('###打印模型model的结构####')
        print(self.networks)
        print('\n')
        print('###打印模型model加载参数前的初始值####')
        print(list(self.networks.parameters()))
        print('\n')
        # 加载预训练参数
        self.load_model_state_dict(self.opt.pretrained_file1, self.opt.pretrained_file2, self.networks)
        print('###打印模型model加载参数后的参数值####')
        print(list(self.networks.parameters()))
        print('\n')
        # 打印各层的requires_grad属性
        print('###打印模型model参数的requires_grad属性####')
        for name, param in self.networks.named_parameters():
            print(name, param.requires_grad)
        """
        # 冻结部分层
        # 将满足条件的参数的 requires_grad 属性设置为False
        for name, value in self.networks.named_parameters():
            if (name != 'SkNet.stage_4') or (name != 'SKNet.skfuse') or (name != "SKNet.gap") or (name != "SKNet.classifier"):
                value.requires_grad = False
        #打印各层的requires_grad属性
        print('###打印模型model参数的requires_grad属性####')
        for name, param in self.networks.named_parameters():
            print(name,param.requires_grad)
        """

    def transfer_state_dict(self, pretrained_dict, model_dict, file):
        # state_dict2 = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        state_dict = {}
        print("file is {}".format(file))
        dict = {"OriginalSingleBranch": "_Ori.", "PrewittSingleBranch": "_Other.", "GaborSingleBranch": "_Other."}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and "SKNet.stage_4" not in k:
                # state_dict.setdefault(k, v)
                state_dict[k] = v
            if "FE" in k:
                temp_k = dict[file].join([k.split(".", 1)[0], k.split(".", 1)[1]])
                if temp_k in model_dict.keys():
                    state_dict[temp_k] = v
                    print("Transfer key(s) in state_dict:{} to {}".format(k, temp_k))
                else:
                    print("Missing key(s) in target_state_dict :{}".format(temp_k))
            if "SKNet" in k:
                temp_k = dict[file].join([k.split(".", 2)[0] + '.' + k.split(".", 2)[1], k.split(".", 2)[2]])
                if temp_k in model_dict.keys():
                    state_dict[temp_k] = v
                    print("Transfer key(s) in state_dict:{} to {}".format(k, temp_k))
                else:
                    print("Missing key(s) in target_state_dict :{}".format(temp_k))
            else:
                print("Missing key(s) in state_dict :{}".format(k))
        return state_dict

    def transfer_model(self, pretrained_file, model):
        pretrained_dict = t.load(pretrained_file)  # get pretrained dict
        model_dict = model.state_dict()  # get model dict
        # 在合并前(update),需要去除pretrained_dict一些不需要的参数
        pretrained_dict = self.transfer_state_dict(pretrained_dict, model_dict, pretrained_file.split("/")[-3])
        model_dict.update(pretrained_dict)  # 更新(合并)模型的参数
        model.load_state_dict(model_dict)
        return model

    def load_model_state_dict(self, pretrained_file1, pretrained_file2, model):
        model = self.transfer_model(pretrained_file1, model)
        model = self.transfer_model(pretrained_file2, model)
        self.networks = model

    def train(self):
        self.load_model_state_dict(self.opt.pretrained_file1, self.opt.pretrained_file2, self.networks)
        self.networks = self.networks.to(self.device)
        if self.opt.ExistsModel is not None:
            print("-- Load pre trained model -- {A}".format(A=self.opt.ExistsModel))
            self.networks.load(self.opt.ExistsModel)
        r"""
        >>> Note: If you want use cudnn to accelerate. You need to handle the trade off
            between performance and speed. In experiment, On 10 set with same config, This
            accelerate method may have (+-)0.8 points trade off on acc. 
        if self.device != "cpu":
            t.backends.cudnn.benchmark = True
            logging.warning("cudnn optimize is open. May spend more time on some networks")
        """
        # -- LOGGING => Data from self.opt - start
        for item in self.opt.__dict__:
            print("{A} ==> {B}".format(A=item, B=self.opt.__dict__[item]))
        # -- LOGGING => Data from self.opt - end
        if self.opt.enable_EMA:
            self.EMA_NET = EMA.EMA(self.networks, 0.999)
            self.EMA_NET.register()
        for epoch in range(self.opt.last_epoch, self.opt.max_epoch):
            print("-- lr = {A}".format(A=self.scheduler.get_last_lr()[0]))
            # -- Dataset reload. Resample by class aware methods
            if epoch < self.opt.FullDatasetTrain:
                base_data, base_loader = run.build_data_loader(self.sampler, self.opt, "Full")
            else:
                base_data, base_loader = run.build_data_loader(self.sampler, self.opt, "Train")
            self.train_base_data = base_data[0]
            self.test_base_data = base_data[1]
            self.train_dataloader = base_loader[0]
            self.test_dataloader = base_loader[1]
            num_each_cls = self.train_base_data.each_cls_num
            # num = self.train_base_data.size  # This line is wrong
            num = sum(num_each_cls)
            # -- LOGGING ==> Dataset info - start
            print("-- Dataset --")
            print("-- Training num ==>      {A}".format(A=self.train_base_data.size))
            print("-- Training num ==>      {A}".format(A=self.train_base_data.each_cls_num))
            print("-- Testing num  ==>      {A}".format(A=self.test_base_data.size))
            print("-- Testing num  ==>      {A}".format(A=self.test_base_data.each_cls_num))
            # -- LOGGING ==> Dataset info - end
            # -- MemData object set zero
            self.MemLoss.set_zero()
            self.MemHammingLoss.set_zero()
            self.MemRankingLoss.set_zero()
            self.MemCoverageLoss.set_zero()
            # -- Start Batch training
            """for idx, (image, gt) in enumerate(self.train_dataloader):
                print(idx, image[0, :].sum(), gt[0, :])
            exit(1)"""
            # self.pbar = tqdm(total=len(self.train_dataloader))
            for idx, (image, gt) in enumerate(self.train_dataloader):
                # self.pbar.update(idx)
                input_image = Variable(image)
                target = Variable(gt)
                input_image = input_image.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                predict = self.networks(input_image)
                # -- Loss function. important! Use each_num_cls
                # loss = self.calculate_loss(predict, target, num_each_cls, num, self.opt.num_cls)
                _weight = t.tensor([1.0 / item for item in num_each_cls]).float().to(self.device)
                _weight = _weight * 1000
                loss = t.nn.MultiLabelSoftMarginLoss(_weight)(predict, target)
                # loss = self.FOCALLOSS(predict, target)
                # -- LOGGING batch loss log - start
                if (idx + 1) % 1000 == 0:
                    print("w: In epoch {A}, batch {B}, loss: {C}".format(
                        A=epoch,
                        B=idx,
                        C=loss.item()
                    ))
                # -- LOGGING batch loss log - end
                hamming_loss = self.calculate_metric(predict, target)
                self.MemLoss.update(loss.item())
                self.MemHammingLoss.update(hamming_loss["HammingLoss"])
                loss.backward()
                self.optimizer.step()
                if self.opt.enable_EMA:
                    self.EMA_NET.update()
            # Close pbar
            # self.pbar.close()
            # TODO save model every 4 epoch
            if (epoch + 1) % self.opt.every_epoch == 0:
                self.save_model(epoch)
            # -- LOGGING ==> Loss - start
            print("-- epoch:      ==> {A}".format(A=epoch))
            print("-- Loss        ==> {A}".format(A=self.MemLoss.mean()))
            print("-- HammingLoss ==> {A}".format(A=self.MemHammingLoss.mean()))
            self.FileLogger.write("-- epoch:      ==> {A}".format(A=epoch))
            self.FileLogger.write("-- Loss        ==> {A}".format(A=self.MemLoss.mean()))
            self.FileLogger.write("-- HammingLoss ==> {A}".format(A=self.MemHammingLoss.mean()))
            self.visual.plot("Training Loss", self.MemLoss.mean())
            self.visual.plot("Training HammingLoss", self.MemHammingLoss.mean())
            self.visual.log("epoch: {epoch}, lr: {lr}, Loss: {loss}".format(
                epoch=epoch,
                lr=self.scheduler.get_last_lr()[0],
                loss=self.MemLoss.mean(),
            ))
            self.val()
            # -- LOGGING ==> Loss - end
            # TODO write Loss to Log file or terminal
            self.MemLoss.set_zero()
            self.MemHammingLoss.set_zero()
            self.scheduler.step(epoch=epoch)

    def GaborFuse_train(self):
        self.load_model_state_dict(self.opt.pretrained_file1, self.opt.pretrained_file2, self.networks)
        self.networks = self.networks.to(self.device)
        if self.opt.ExistsModel is not None:
            print("-- Load pre trained model -- {A}".format(A=self.opt.ExistsModel))
            self.networks.load(self.opt.ExistsModel)
        r"""
        >>> Note: If you want use cudnn to accelerate. You need to handle the trade off
            between performance and speed. In experiment, On 10 set with same config, This
            accelerate method may have (+-)0.8 points trade off on acc. 
        if self.device != "cpu":
            t.backends.cudnn.benchmark = True
            logging.warning("cudnn optimize is open. May spend more time on some networks")
        """
        # -- LOGGING => Data from self.opt - start
        for item in self.opt.__dict__:
            print("{A} ==> {B}".format(A=item, B=self.opt.__dict__[item]))
        # -- LOGGING => Data from self.opt - end
        if self.opt.enable_EMA:
            self.EMA_NET = EMA.EMA(self.networks, 0.999)
            self.EMA_NET.register()
        for epoch in range(self.opt.last_epoch, self.opt.max_epoch):
            print("-- lr = {A}".format(A=self.scheduler.get_last_lr()[0]))
            # -- Dataset reload. Resample by class aware methods
            if epoch < self.opt.FullDatasetTrain:
                base_data, base_loader = run.build_data_loader(self.sampler, self.opt, "Full")
            else:
                base_data, base_loader = run.build_data_loader(self.sampler, self.opt, "Train")
            self.train_base_data = base_data[0]
            self.test_base_data = base_data[1]
            self.train_dataloader = base_loader[0]
            self.test_dataloader = base_loader[1]
            num_each_cls = self.train_base_data.each_cls_num
            # num = self.train_base_data.size  # This line is wrong
            num = sum(num_each_cls)
            # -- LOGGING ==> Dataset info - start
            print("-- Dataset --")
            print("-- Training num ==>      {A}".format(A=self.train_base_data.size))
            print("-- Training num ==>      {A}".format(A=self.train_base_data.each_cls_num))
            print("-- Testing num  ==>      {A}".format(A=self.test_base_data.size))
            print("-- Testing num  ==>      {A}".format(A=self.test_base_data.each_cls_num))
            # -- LOGGING ==> Dataset info - end
            # -- MemData object set zero
            self.MemLoss.set_zero()
            self.MemHammingLoss.set_zero()
            self.MemRankingLoss.set_zero()
            self.MemCoverageLoss.set_zero()
            # -- Start Batch training
            """for idx, (image, gt) in enumerate(self.train_dataloader):
                print(idx, image[0, :].sum(), gt[0, :])
            exit(1)"""
            # self.pbar = tqdm(total=len(self.train_dataloader))
            for idx, (image, Gabor_image, gt) in enumerate(self.train_dataloader):
                # self.pbar.update(idx)
                input_image = Variable(image)
                input_Gabor_image = Variable(Gabor_image)
                target = Variable(gt)
                input_image = input_image.to(self.device)
                input_Gabor_image = input_Gabor_image.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                predict = self.networks(input_image, input_Gabor_image)
                # -- Loss function. important! Use each_num_cls
                # loss = self.calculate_loss(predict, target, num_each_cls, num, self.opt.num_cls)
                _weight = t.tensor([1.0 / item for item in num_each_cls]).float().to(self.device)
                _weight = _weight * 1000
                loss = t.nn.MultiLabelSoftMarginLoss(_weight)(predict, target)
                # loss = self.FOCALLOSS(predict, target)
                # -- LOGGING batch loss log - start
                if (idx + 1) % 1000 == 0:
                    print("w: In epoch {A}, batch {B}, loss: {C}".format(
                        A=epoch,
                        B=idx,
                        C=loss.item()
                    ))
                # -- LOGGING batch loss log - end
                hamming_loss = self.calculate_metric(predict, target)
                self.MemLoss.update(loss.item())
                self.MemHammingLoss.update(hamming_loss["HammingLoss"])
                loss.backward()
                self.optimizer.step()
                if self.opt.enable_EMA:
                    self.EMA_NET.update()
            # Close pbar
            # self.pbar.close()
            # TODO save model every 4 epoch
            if (epoch + 1) % self.opt.every_epoch == 0:
                self.save_model(epoch)
            # -- LOGGING ==> Loss - start
            print("-- epoch:      ==> {A}".format(A=epoch))
            print("-- Loss        ==> {A}".format(A=self.MemLoss.mean()))
            print("-- HammingLoss ==> {A}".format(A=self.MemHammingLoss.mean()))
            self.FileLogger.write("-- epoch:      ==> {A}".format(A=epoch))
            self.FileLogger.write("-- Loss        ==> {A}".format(A=self.MemLoss.mean()))
            self.FileLogger.write("-- HammingLoss ==> {A}".format(A=self.MemHammingLoss.mean()))
            self.visual.plot("Training Loss", self.MemLoss.mean())
            self.visual.plot("Training HammingLoss", self.MemHammingLoss.mean())
            self.visual.log("epoch: {epoch}, lr: {lr}, Loss: {loss}".format(
                epoch=epoch,
                lr=self.scheduler.get_last_lr()[0],
                loss=self.MemLoss.mean(),
            ))
            self.GaborFuse_val()
            # -- LOGGING ==> Loss - end
            # TODO write Loss to Log file or terminal
            self.MemLoss.set_zero()
            self.MemHammingLoss.set_zero()
            self.scheduler.step(epoch=epoch)

    @staticmethod
    def multi_cls_acc(predict, label):
        print("--------Multi-Label-Cls-ACC--------")
        print(len(predict), len(label))
        print("------------self-compare-----------")
        NumCls = len(predict[0])
        for ii in range(0, NumCls):
            TP = FP = TN = FN = NUM_LABLE_P = NUM_LABLE_N = 0
            for idx in range(0, len(predict)):
                if predict[idx][ii] == 1 and label[idx][ii] == 1:
                    TP += 1
                elif predict[idx][ii] == 0 and label[idx][ii] == 0:
                    TN += 1
                elif predict[idx][ii] == 1 and label[idx][ii] == 0:
                    FP += 1
                elif predict[idx][ii] == 0 and label[idx][ii] == 1:
                    FN += 1

                if label[idx][ii] == 1:
                    NUM_LABLE_P += 1
                elif label[idx][ii] == 0:
                    NUM_LABLE_N += 1
            recall = TP / (TP + FN)
            precision = TP / (TP + FP)
            f1 = (2 * precision * recall) / (precision + recall)
            print(
                "Cls idx: {ClsIdx}, Accuracy: {ClsAcc}, TP:{TP}, TN:{TN}, FP:{FP}, FN:{FN}, NUMP:{NUM_P}, NUM_N:{NUM_N}".format(
                    ClsIdx=ii, ClsAcc=(TP + TN) * 100 / (TP + FN + TN + FP),
                    TP=TP, TN=TN, FP=FP, FN=FN, NUM_P=NUM_LABLE_P,
                    NUM_N=NUM_LABLE_N
                ))
            print("-----------------------------------------")
            print("Cls idx: {ClsIdx}, F1: {f1}".format(ClsIdx=ii, f1=f1))

    def test(self, network):
        base_data, base_loader = run.build_data_loader(self.sampler, self.opt, "Full")
        self.test_base_data = base_data[1]
        self.test_dataloader = base_loader[1]
        PREDICT = []
        TARGET = []
        NET = network
        if self.opt.val_ema_model is not None:
            # when use Prewitt, the parameters will store the class name due to separate
            # implementation, so u need to use nn.DataParallel(model) to process it again.
            load_model(NET, self.opt.val_ema_model, self.device, True)
        else:
            load_model(NET, self.opt.val_model, self.device, True)
        with t.no_grad():
            NET.eval()
            print(len(self.test_base_data))
            for idx, (image, target) in enumerate(self.test_dataloader):
                if (idx + 1) % 100 == 1:
                    logging.info("Run on idx: {A}".format(A=idx))
                image = Variable(image).to(self.device)
                target = Variable(target).to(self.device)
                predict = NET(image)
                # the predict score has already usr sigmoid function to process.
                predict = np.where((predict.detach().cpu().numpy()) > 0.5, 1, 0).tolist()[0]
                label = target.cpu().detach().numpy().astype("int").tolist()[0]
                PREDICT.append(predict)
                TARGET.append(label)
        run.multi_cls_acc(PREDICT, TARGET)

    def GaborFuse_test(self, network):
        base_data, base_loader = run.build_data_loader(self.sampler, self.opt, "Full")
        self.test_base_data = base_data[1]
        self.test_dataloader = base_loader[1]
        PREDICT = []
        TARGET = []
        NET = network
        if self.opt.val_ema_model is not None:
            # when use Prewitt, the parameters will store the class name due to separate
            # implementation, so u need to use nn.DataParallel(model) to process it again.
            load_model(NET, self.opt.val_ema_model, self.device, False)
        else:
            load_model(NET, self.opt.val_model, self.device, False)
        with t.no_grad():
            NET.eval()
            print(len(self.test_base_data))
            for idx, (image, Gabor_image, target) in enumerate(self.test_dataloader):
                if (idx + 1) % 100 == 1:
                    logging.info("Run on idx: {A}".format(A=idx))
                image = Variable(image).to(self.device)
                Gabor_image = Variable(Gabor_image).to(self.device)
                target = Variable(target).to(self.device)
                predict = NET(image, Gabor_image)
                # the predict score has already usr sigmoid function to process.
                predict = np.where((predict.detach().cpu().numpy()) > 0.5, 1, 0).tolist()[0]
                label = target.cpu().detach().numpy().astype("int").tolist()[0]
                PREDICT.append(predict)
                TARGET.append(label)
        run.multi_cls_acc(PREDICT, TARGET)

    def val(self):
        self.networks.eval()
        if self.opt.enable_EMA:
            self.EMA_NET.model.eval()
        with t.no_grad():
            _val_bce_loss = MemData()
            _val_hamming_loss = MemData()
            _val_bce_loss.set_zero()
            _val_hamming_loss.set_zero()
            for idx, (image, target) in enumerate(self.test_dataloader):
                image = Variable(image).to(self.device)
                target = Variable(target).to(self.device)
                predict = self.networks(image)
                # -- metric and BCE loss.
                hamming_loss = run.calculate_metric(predict, target)["HammingLoss"]
                _val_hamming_loss.update(hamming_loss)
                bce_loss = t.nn.MultiLabelSoftMarginLoss(weight=None)(predict, target).item()
                _val_bce_loss.update(bce_loss)
        self.visual.plot("Val BCE loss", _val_bce_loss.mean())
        self.visual.plot("Val hamming loss", _val_hamming_loss.mean())
        self.FileLogger.write("Val BCE loss: {A}".format(A=_val_bce_loss.mean()))
        self.FileLogger.write("Val hamming loss: {A}".format(A=_val_hamming_loss.mean()))
        self.networks.train()
        if self.opt.enable_EMA:
            self.EMA_NET.model.train()

    def GaborFuse_val(self):
        self.networks.eval()
        if self.opt.enable_EMA:
            self.EMA_NET.model.eval()
        with t.no_grad():
            _val_bce_loss = MemData()
            _val_hamming_loss = MemData()
            _val_bce_loss.set_zero()
            _val_hamming_loss.set_zero()
            for idx, (image, Gabor_image, target) in enumerate(self.test_dataloader):
                image = Variable(image).to(self.device)
                Gabor_image = Variable(Gabor_image).to(self.device)
                target = Variable(target).to(self.device)
                predict = self.networks(image, Gabor_image)
                # -- metric and BCE loss.
                hamming_loss = run.calculate_metric(predict, target)["HammingLoss"]
                _val_hamming_loss.update(hamming_loss)
                bce_loss = t.nn.MultiLabelSoftMarginLoss(weight=None)(predict, target).item()
                _val_bce_loss.update(bce_loss)
        self.visual.plot("Val BCE loss", _val_bce_loss.mean())
        self.visual.plot("Val hamming loss", _val_hamming_loss.mean())
        self.FileLogger.write("Val BCE loss: {A}".format(A=_val_bce_loss.mean()))
        self.FileLogger.write("Val hamming loss: {A}".format(A=_val_hamming_loss.mean()))
        self.networks.train()
        if self.opt.enable_EMA:
            self.EMA_NET.model.train()

    def test_env(self):
        # -- Loss Test.

        pass
