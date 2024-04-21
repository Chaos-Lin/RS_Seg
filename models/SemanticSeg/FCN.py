# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
import torch.nn.functional as F

import torch

from ..subnet import ConvBNReLU


class FCN(nn.Module):
    """
    A simple implementation for FCN based on PaddlePaddle.

    The original article refers to
    Evan Shelhamer, et, al. "Fully Convolutional Networks for Semantic Segmentation"
    (https://arxiv.org/abs/1411.4038).

    Args:
        num_classes (int): The unique number of target classes.
        backbone (paddle.nn.Module): Backbone networks.
        backbone_indices (tuple, optional): The values in the tuple indicate the indices of output of backbone.
            Default: (-1, ).
        channels (int, optional): The channels between conv Module and the last Module of FCNHead.
            If None, it will be the number of channels of input features. Default: None.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.  Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=(-1, ),
                 channels=None,
                 align_corners=False,
                 pretrained=None,
                 bias=True,
                 data_format="NCHW"):
        super(FCN, self).__init__()

        if data_format != 'NCHW':
            raise ('fcn only support NCHW data format')
        self.backbone = backbone
        backbone_channels = [
            backbone.feat_channels[i] for i in backbone_indices
        ]

        self.head = FCNHead(
            num_classes,
            backbone_indices,
            backbone_channels,
            channels,
            bias=bias)

        self.align_corners = align_corners
        self.pretrained = pretrained
        self.data_format = data_format
        self.init_weight()

    def forward(self, x):
        feat_list = self.backbone(x)
        logit_list = self.head(feat_list)
        return [
            F.interpolate(
                logit,
                torch.shape(x)[2:],
                mode='bilinear',
                align_corners=self.align_corners) for logit in logit_list
        ]

    def init_weight(self):
        if self.pretrained is not None:
            pass
            # utils.load_entire_model(self, self.pretrained)


class FCNHead(nn.Module):
    """
    A simple implementation for FCNHead based on PaddlePaddle

    Args:
        num_classes (int): The unique number of target classes.
        backbone_indices (tuple, optional): The values in the tuple indicate the indices of output of backbone.
            Default: (-1, ).
        channels (int, optional): The channels between conv layer and the last layer of FCNHead.
            If None, it will be the number of channels of input features. Default: None.
        pretrained (str, optional): The path of pretrained model. Default: None
    """

    def __init__(self,
                 num_classes,
                 backbone_indices=(-1, ),
                 backbone_channels=(270, ),
                 channels=None,
                 bias=True):
        super(FCNHead, self).__init__()

        self.num_classes = num_classes
        self.backbone_indices = backbone_indices
        if channels is None:
            channels = backbone_channels[0]

        self.conv_1 = ConvBNReLU(
            in_channels=backbone_channels[0],
            out_channels=channels,
            kernel_size=1,
            stride=1,
            bias=bias)
        self.cls = nn.Conv2d(
            in_channels=channels,
            out_channels=self.num_classes,
            kernel_size=1,
            stride=1,
            bias=bias)
        self.init_weight()

    def forward(self, feat_list):
        logit_list = []
        x = feat_list[self.backbone_indices[0]]
        x = self.conv_1(x)
        logit = self.cls(x)
        logit_list.append(logit)
        return logit_list

    def init_weight(self):
        for name, sublayer in self.named_modules():
            if isinstance(sublayer, nn.Conv2d):
                # 对Conv2d层进行初始化
                nn.init.kaiming_normal_(sublayer.weight, mode='fan_out', nonlinearity='relu')
                if sublayer.bias is not None:
                    nn.init.constant_(sublayer.bias, 0)
            elif isinstance(sublayer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
                # 对BatchNorm1d或SyncBatchNorm层进行初始化
                nn.init.constant_(sublayer.weight, 1)
                nn.init.constant_(sublayer.bias, 0)


