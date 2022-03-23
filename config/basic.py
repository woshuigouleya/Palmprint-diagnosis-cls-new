import os
import sys
from pathlib import Path


class conf:
    def __init__(self):
        # -- Dataset 44 class
        """
        self.NeedCls = [[1, 1], [1, 2], [1, 3], [1, 4],
                        [2, 1], [2, 2], [2, 3],
                        [3, 1], [3, 2], [3, 3], [3, 4], [3, 5],
                        [4, 1], [4, 2], [4, 3], [4, 4],
                        [5, 1], [5, 2],
                        [6, 1], [6, 2],
                        [7, 1], [7, 2], [7, 3],
                        [8, 1], [8, 2], [8, 3], [8, 4], [8, 5],
                        [9, 1], [9, 2], [9, 3], [9, 4], [9, 5], [9, 6], [9, 7], [9, 8],
                        [10, 1], [10, 2], [10, 3], [10, 4], [10, 5], [10, 6], [10, 7], [10, 8],
                        ]
        self.each_cls_num = [3239, 1451, 13847, 4681, 1446, 2021, 1036, 2683, 5615, 6448, 1136, 4136, 3290, 5643, 1273,
                             4601, 1172, 1891, 3041, 2292, 7545, 3110, 9191, 599, 6316, 2239, 7636, 1154, 2236, 1083,
                             2506, 2616, 2594, 2139, 11329, 660, 937, 73, 958, 438, 3107, 1134, 556, 1790]
        self.total_num = 34732  # Used only in debug mode
        """
        # -- Dataset 24 class
        self.NeedCls = [[4, 3], [2, 1], [1, 2], [10, 8], [5, 2], [2, 2], [9, 6], [8, 3],
                        [9, 1], [6, 2], [9, 3], [9, 5], [9, 4], [3, 1], [6, 1], [7, 2], [10, 5],
                        [1, 1], [4, 1], [3, 5], [4, 4], [1, 4], [3, 2],
                        [4, 2]]
        self.each_cls_num = [1024, 1165, 1179, 1462, 1532, 1635, 1740, 1801, 1810, 1858, 1997, 2099, 2131, 2157, 2460,
                             2478, 2501, 2576, 2670, 3342, 3715, 3804, 4533, 4547]
        self.total_num = sum(self.each_cls_num)
        # -- Basic.
        self.index_path = "/home/liu/gyq/index-cls4train.csv"
        self.rate = 0.8  # TODO 0.8 by default
        # TODO n_max_sampler 2 in test. Default by 2000
        self.n_max_sampler = 256  # TODO n_max for class-aware sampler. Basic set n_max // 4
        # -- Data loader
        self.batch_size_train = 80
        self.batch_size_test = 10
        self.shuffle_train = True
        self.shuffle_test = False
        self.num_workers_train = 4
        self.num_workers_test = 4
        # -- Optimizer SGD option
        self.lr = 0.02
        self.momentum = 0.9
        self.weight_decay = 1e-4
        # -- scheduler
        self.milestones = [60, 120, 180]
        self.scheduler_gamma = 0.1
        self.last_epoch = 0  # it will be used in fine-tune period. -1 means training from head.
        # -- Train
        self.num_cls = len(self.NeedCls)
        self.max_epoch = 800
        self.enable_EMA = True
        self.every_epoch = 4  # TODO normally set to 4
        self.WarmUpEpoch = 4  # TODO normally set to 4
        self.FullDatasetTrain = 800  # TODO normally set to 60
        self.ExistsModel = None
        # -- DB loss
        self.DB_alpha = 0.1
        self.DB_miu = 0.2
        self.DB_lambda = 2.0
        self.DB_beta = 10.0
        self.DB_k = 0.05
        # -- Focal loss
        self.FO_alpha = 0.25
        self.FO_gamma = 2.0
        # -- work shop
        self.main_path = "/home/liu/gyq/workshop/FuseGaborBranch/"
        # Test config
        self.val_model = None
        self.val_ema_model = None
        self.pretrained_file1 = "/home/liu/gyq/workshop/OriginalSingleBranch/ema_model/Epoch107.pth"
        self.pretrained_file2 = "/home/liu/gyq/workshop/GaborSingleBranch/ema_model/Epoch107.pth"
        """
        Model of Prewitt single branch. using sknet
        Run on machine 54_xxl_413 Epoch->103.
        Test file stored in /workspace/PrewittSingleBranch/log/test_2022_1_19.md
        self.val_model = "/home/liu/gyq/workshop/PrewittSingleBranch/ori_model/Epoch103.pth"
        self.val_ema_model = "/home/liu/gyq/workshop/PrewittSingleBranch/ema_model/Epoch103.pth"
        """
        """
        Model of basic sknet-> Epoch163 -> num_cls 23 ->
        Run on machine 129_xxl_413
        # self.val_model = "/media/liu/WchDisk/workshop/palm_lt/train4/ori_model/Epoch163.pth"
        # self.val_ema_model = "/media/liu/WchDisk/workshop/palm_lt/train4/ema_model/Epoch163.pth"
        """
        """
        Model of GaborFilter single branch. using sknet.
        Run on machine 54_xxl_413 Epoch->107.
        Test file stored in /workspace/GaborSingleBranch/log/test_2022_1_17.md
        """
        self.val_model = "/home/liu/gyq/workshop/PrewittSingleBranch/ori_model/Epoch99.pth"
        self.val_ema_model = "/home/liu/gyq/workshop/PrewittSingleBranch/ema_model/Epoch99.pth"
