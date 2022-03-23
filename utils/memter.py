class MemData:
    def __init__(self):
        self.__data__ = 0.0
        self.__counter__ = 0

    def update(self, rhs):
        self.__data__ = self.__data__ + rhs
        self.__counter__ = self.__counter__ + 1

    def set_zero(self):
        self.__data__ = 0.0
        self.__counter__ = 0

    def mean(self):
        return self.__data__ / self.__counter__
