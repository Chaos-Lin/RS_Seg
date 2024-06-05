"""
AMIO -- All Model in One
"""
import torch.nn as nn

from .Classification import MobileViT
from .SemanticSeg import *
from .Classification import *
# from .SemanticSeg.UNet import UNet
# from .multiTask import *
# from .singleTask import *
# from .subNets import AlignSubNet


class AMIO(nn.Module):
    def __init__(self, args):
        super(AMIO, self).__init__()
        self.MODEL_MAP = {

            'unet': UNet,
            'unetplusplus': UNetPlusPlus,
            'u2net': U2Net,
            'segnet': SegNet,
            'fcn': FCN,
            'encnet': ENCNet,
            'deeplabv3': DeepLabV3,
            'deeplabv3p': DeepLabV3P,
            "bisenetv1":BiSeNetV1,
            "bisenetv2":BiSeNetV2,
            # 'unetp':UnetP

            'mobilevit': MobileViT,
            'alexnet': AlexNet


        }
        lastModel = self.MODEL_MAP[args['model_name']]
        # print(lastModel)
    #     # 获取的模型类
        self.Model = lastModel(args)
    #     # 创建了一个模型实例
    #
    def forward(self,x, *args, **kwargs):
        return self.Model(x, *args, **kwargs)
