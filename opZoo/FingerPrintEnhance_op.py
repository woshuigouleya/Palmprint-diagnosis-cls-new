import os
import sys
import fingerprint_enhancer

sys.path.append(os.path.split(sys.path[0])[0])


class FingerPrintMethod:
    @staticmethod
    def Enhance(images):
        return fingerprint_enhancer.enhance_Fingerprint(images)
