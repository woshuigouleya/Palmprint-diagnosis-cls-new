import numpy as np
import cv2


class GaborMethod:
    @staticmethod
    def build_filters():
        filters = []
        ksize = 128
        for theta in np.arange(0, np.pi, np.pi / 16):
            # 4.0, theta, 10.0, 0.5, 0
            kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5 * kern.sum()
            filters.append(kern)
        return filters

    @staticmethod
    def process(img, filters):
        accum = np.zeros_like(img)
        for kern in filters:
            fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
            np.maximum(accum, fimg, accum)
        return accum

    @staticmethod
    def Enhance(image):
        filters = GaborMethod.build_filters()
        return GaborMethod.process(image, filters)
