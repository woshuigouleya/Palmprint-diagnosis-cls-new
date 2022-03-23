import os
import sys
import random
import logging
from pathlib import Path

current_folder = Path(__file__).absolute().parent
os.chdir(str(current_folder))
sys.path.append(str(current_folder.parent))

from sampler import get_data


class CWSampler:
    def __init__(self, index_path, wash, rate):
        self.index_path = index_path
        self.wash = wash
        self.rate = rate
        self.num_cls = len(wash)
        self.inner_data = get_data.InnerData(self.index_path, self.wash, self.rate)
        self.n_max = max(self.inner_data.each_cls_num) // 4
        self.__training_data = get_data.Data(self.inner_data.data("Train"), self.num_cls)
        self.__test_data = get_data.Data(self.inner_data.data("Test"), self.num_cls)

    def step(self, n_max=None):
        if n_max is not None:
            self.n_max = n_max
        else:
            logging.warning("In CWsampler, n_max is max(...)//2 by default.")
        ans = []
        for ii in range(self.num_cls):
            num = self.inner_data.each_cls_num[ii] - 1
            for ith in range(self.n_max):
                ans.append(self.inner_data.slice_index[ii][random.randint(0, num)])
        ans_data = get_data.Data(ans, self.num_cls)
        self.__training_data = ans_data

    # @property
    def data(self, rhs="Train"):
        if rhs == "Train":
            return self.__training_data
        elif rhs == "Test":
            return self.__test_data
        else:
            logging.error(rhs, " is not in data opt")
