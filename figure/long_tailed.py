import os
import sys
from pathlib import Path
from matplotlib import pyplot as plt

current_folder = Path(__file__).absolute().parent
os.chdir(str(current_folder))
sys.path.append(str(current_folder.parent))

from sampler import get_data, CWsampler
from config.basic import conf


def draw_long_tailed_fig_ori(_opt, save_path):
    buf = get_data.InnerData(_opt.index_path, _opt.NeedCls)
    plt.bar(range(len(buf.each_cls_num)), sorted(buf.each_cls_num))
    plt.savefig(save_path)


def draw_fig_class_ware(_opt, save_path):
    sam = CWsampler.CWSampler(_opt.index_path, _opt.NeedCls)
    new_training_data = sam.step()
    buf = new_training_data.each_cls_num
    plt.bar(range(len(buf)), sorted(buf))
    plt.savefig(save_path)


if __name__ == "__main__":
    opt = conf()
    buf = get_data.InnerData(opt.index_path, opt.NeedCls, 0.8)
    print(buf.each_cls_num)
    # draw_long_tailed_fig_ori(opt, './DATA/original_data.png')
    # draw_fig_class_ware(opt, 'class_ware_sampled.png')
    exit(1)
