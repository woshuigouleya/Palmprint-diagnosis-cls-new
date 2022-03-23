import torch as t
import logging

import core
import net
import cv2
from config.basic import conf
from netZoo.Gabor_SingleBranch import GaborSingleBranch
from netZoo.Prewitt_SingleBranch import PrewittSingleBranch
from netZoo.Original_SingleBranch import OriginalSingleBranch
from netZoo.Fuse_PrewittBranch import FusePrewittBranch
from netZoo.Fuse_GaborBranch import FuseGaborBranch

if __name__ == "__main__":
    """
    RUN = core.run(net.MLLTnet(option.num_cls), option)
    RUN.train()
    RUN = core.run(net.MLLTnet(option.num_cls), option)
    RUN.test(net.MLLTnet(option.num_cls))
    """

    """
    RUN = core.run(PrewittSingleBranch(option.num_cls), option)
    RUN.train()
    """

    option = conf()
    RUN = core.run(PrewittSingleBranch(option.num_cls), option)
    RUN.test(PrewittSingleBranch(option.num_cls))


    """
    option = conf()
    RUN = core.run(GaborSingleBranch(option.num_cls), option)
    RUN.train()
    """
    """
    option = conf()
    RUN = core.run(GaborSingleBranch(option.num_cls), option)
    RUN.test(GaborSingleBranch(option.num_cls))
    """

    """
    option = conf()
    RUN = core.run(OriginalSingleBranch(option.num_cls), option)
    RUN.train()
    """
    """
    option = conf()
    RUN = core.run(OriginalSingleBranch(option.num_cls), option)
    RUN.test(OriginalSingleBranch(option.num_cls))
    """

    """
    option = conf()
    RUN = core.run(FusePrewittBranch(option.num_cls), option)
    # RUN.print_model()
    RUN.train()
    """
    """
    option = conf()
    RUN = core.run(FusePrewittBranch(option.num_cls), option)
    RUN.test(FusePrewittBranch(option.num_cls))
    """
    """
    option = conf()
    RUN = core.run(FuseGaborBranch(option.num_cls), option)
    # RUN.print_model()
    RUN.GaborFuse_train()
    """
    """
    option = conf()
    RUN = core.run(FuseGaborBranch(option.num_cls), option)
    RUN.GaborFuse_test(FuseGaborBranch(option.num_cls))
    """



