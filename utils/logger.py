import os
import sys
import logging


class FileLogger:
    def __init__(self, save_path):
        self.save_path = save_path

    def write(self, rhs, mode="INFO"):
        if mode not in ["INFO", "WARN", "ERR"]:
            logging.error("FileLogger not support mode !")
        # rhs is a string
        f = open(self.save_path, "a")
        pre_string = ">>" + mode + ": "
        f.write(pre_string + rhs + "\n")
        f.close()
