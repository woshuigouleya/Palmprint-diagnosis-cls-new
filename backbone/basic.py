import time
import torch as t


class BasicModule(t.nn.Module):
    """
    封装nn.Module,提供save and load
    """

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        """
        load exist model
        :param path:
        :return: None
        """
        self.load_state_dict(t.load(path))

    def save(self, path, name=None):
        """
        save model file
        :param name:
        :param path:
        :return: None
        """
        if name is None:
            prefix = path + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%S.pth')
        t.save(self.state_dict(), name)
        return name
