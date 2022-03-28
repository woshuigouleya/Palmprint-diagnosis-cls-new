import os
import sys
import csv
import logging
from pathlib import Path
from itertools import islice

current_folder = Path(__file__).absolute().parent
os.chdir(str(current_folder))
sys.path.append(str(current_folder.parent))


class InnerData:
    def __init__(self, index_path, wash, rate=None):
        assert (os.path.exists(index_path))
        if rate is None:
            logging.warning("training / test / val dataset not split.")
        self.index_path = index_path
        self.wash = wash
        self.rate = rate
        self.num_cls = len(wash)
        # -- original data.
        # load in list[dic, dic,...], dic={image_path, GT}
        self._data = []
        self.phase_data()  # function
        self.size = len(self._data)
        self.each_cls_num = self.get_each_cls_num()  # in train.
        self.slice_index = self.get_slice_index()  # in train.

    def phase_data(self):
        _f = open(self.index_path, "r")
        _f_reader = csv.reader(_f)
        for item in islice(_f_reader, 1, None):
            img_path = item[0]
            gt = InnerData.phase_string(item[1], self.wash)
            self._data.append({'index': img_path, 'gt': gt})
        # random.shuffle(self._data)
        # TODO under this block, need to be used?
        """
        for item in self._data:
            assert (sum(item['gt']) > 0)
        """

    def getitem(self, index):
        return self._data[index]

    def get_each_cls_num(self):
        ans = []
        for i in range(self.num_cls):
            ans.append(0)
        for i in range(int(self.rate * self.size)):
            for idx in range(self.num_cls):
                if self._data[i]['gt'][idx] == 1:
                    ans[idx] += 1
        return ans

    def get_slice_index(self):
        ans = []
        for i in range(self.num_cls):
            ans.append([])
        for idx in range(int(self.rate * self.size)):
            for j in range(self.num_cls):
                if self._data[idx]['gt'][j] == 1:
                    ans[j].append(self._data[idx])
        return ans

    @staticmethod
    def phase_string(rhs, wash):
        num = len(wash)
        ans = []
        for idx in range(0, num):
            ans.append(0)
        buf = rhs.split(';')
        for item in buf:
            jug = item.split(',')
            jug = [int(item) for item in jug]
            for idx in range(0, num):
                if wash[idx] == jug:
                    ans[idx] = 1
        return ans

    # @property
    def data(self, rhs="Train"):
        if rhs == "Train":
            return self._data[:int(self.rate * self.size)]
        elif rhs == "Test":
            return self._data[int(self.rate * self.size):]
        else:
            logging.error(rhs, " is not in data opt")


class Data:
    """
    You can not change the _rhs in Data class.
    """

    def __init__(self, rhs, num_cls):
        self.__rhs = rhs
        self.size = len(self.__rhs)
        self.num_cls = num_cls
        self.each_cls_num = self.get_each_cls_num()

    def getitem(self, index):
        return self.__rhs[index]

    def get_each_cls_num(self):
        ans = []
        for i in range(self.num_cls):
            ans.append(0)
        for i in range(self.size):
            for idx in range(self.num_cls):
                if self.__rhs[i]['gt'][idx] == 1:
                    ans[idx] += 1
        return ans

    def __getitem__(self, item):
        return self.__rhs[item]

    def __len__(self):
        return self.size

    # @property
    def data(self):
        return self.__rhs
