import logging
import os


class WorkManager:
    def __init__(self, main_path):
        if not os.path.exists(main_path):
            logging.error("WorkManager can't find proper path to save experiment data.")
            exit(1)
        self.MainPath = main_path
        self.LogPath = os.path.join(self.MainPath, "log")
        if not os.path.exists(self.LogPath):
            os.mkdir(self.LogPath)
        self.ori_model_path = os.path.join(self.MainPath, "ori_model")
        if not os.path.exists(self.ori_model_path):
            os.mkdir(self.ori_model_path)
        self.ema_model_path = os.path.join(self.MainPath, "ema_model")
        if not os.path.exists(self.ema_model_path):
            os.mkdir(self.ema_model_path)
