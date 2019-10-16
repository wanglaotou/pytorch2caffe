# -*- coding: UTF-8 -*-
'''
 * @Author: Jiangtao
 * @Email: jiangtaoo2333@163.com
 * @Company: Streamax
 * @Date: 2019/08/05 14:03
 * @Description: 
'''
import logging as log
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._basic_layer import *



class backBone(nn.Module):
    def __init__(self):
        super(backBone, self).__init__()

        layer0 = Conv2dBatchReLU(1, 32, 3, 2)       #64
        layer1 = Conv2dBatchReLU(32, 64, 3, 1)
        layer2 = Conv2dBatchReLU(64, 64, 3, 2)      #32
        layer3 = Conv2dBatchReLU(64, 128, 3, 1)
        layer4 = Conv2dBatchReLU(128, 128, 3, 2)    #16

        self.layers = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
            layer4
        )

    def forward(self, x):

        x = self.layers(x)

        return x

class eyeBone(nn.Module):
    def __init__(self):
        super(eyeBone, self).__init__()

        layer0 = Conv2dBatchReLU(128, 128, 3, 1)
        layer1 = Conv2dBatchReLU(128, 256, 3, 2)
        layer2 = Conv2dBatchReLU(256, 512, 3, 2)
        layer3 = GlobalAvgPool2d()
        layer4 = FullyConnectLayer(512, 8)

        self.layers = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
            layer4
        )

    def forward(self, x):

        x = self.layers(x)

        return x

class mouthBone(nn.Module):
    def __init__(self):
        super(mouthBone, self).__init__()

        layer0 = Conv2dBatchReLU(128, 128, 3, 1)
        layer1 = Conv2dBatchReLU(128, 256, 3, 2)
        layer2 = Conv2dBatchReLU(256, 512, 3, 2)
        layer3 = GlobalAvgPool2d()
        layer4 = FullyConnectLayer(512, 16)

        self.layers = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
            layer4
        )

    def forward(self, x):

        x = self.layers(x)

        return x


class faceBone(nn.Module):
    def __init__(self):
        super(faceBone, self).__init__()

        layer0 = Conv2dBatchReLU(128, 128, 3, 1)
        layer1 = Conv2dBatchReLU(128, 256, 3, 2)
        layer2 = Conv2dBatchReLU(256, 512, 3, 2)
        layer3 = GlobalAvgPool2d()
        
        layer4 = FullyConnectLayer(512, 10)
        layer5 = FullyConnectLayer(10, 4)

        self.features = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
        )
        
        self.classifier = nn.Sequential(
            layer4,
            layer5,
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x      

class detectBone(nn.Module):
    def __init__(self):
        super(detectBone, self).__init__()

        layer0 = Conv2dBatchReLU(128, 128, 3, 1)
        layer1 = Conv2dBatchReLU(128, 256, 3, 2)
        layer2 = Conv2dBatchReLU(256, 512, 3, 2)
        layer3 = GlobalAvgPool2d()
        layer4 = FullyConnectLayer(512, 4)

        self.layers = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
            layer4
        )

    def forward(self, x):

        x = self.layers(x)

        return x

class emotionBone(nn.Module):
    def __init__(self):
        super(emotionBone, self).__init__()

        layer0 = Conv2dBatchReLU(128, 128, 3, 1)
        layer1 = Conv2dBatchReLU(128, 256, 3, 2)
        layer2 = Conv2dBatchReLU(256, 512, 3, 2)
        layer3 = GlobalAvgPool2d()
        layer4 = FullyConnectLayer(512, 2)

        self.layers = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
            layer4
        )

    def forward(self, x):

        x = self.layers(x)

        return x