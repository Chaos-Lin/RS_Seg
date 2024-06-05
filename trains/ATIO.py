"""
ATIO -- All Trains in One
"""
from .SemanticSeg import *
from .Classification import *


__all__ = ['ATIO']


class ATIO():
    def __init__(self):
        self.TRAIN_MAP = {
            'unet':UNet,
            'unetplusplus':UNet,
            'u2net':UNet,
            'segnet':UNet,
            'unetp':UNet,
            "bisenetv1": BiseNet,
            "bisenetv2": BiseNet,

            'alexnet':AlexNet,




        }

    def getTrain(self, args):
        return self.TRAIN_MAP[args['model_name']](args)
