'''
@Author: Jiangtao
@Date: 2019-08-29 15:43:40
@LastEditors: Jiangtao
@LastEditTime: 2019-08-29 17:02:15
@Description: 
'''

import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import layer

import caffe

print(caffe.__path__)

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, c, h, w].
        Returns:
            a float tensor with shape [batch_size, c*h*w].
        """

        # without this pretrained model isn't working
        x = x.transpose(3, 2).contiguous()

        return x.view(x.size(0), -1)

class PNet(nn.Module):

    def __init__(self, is_train=False):
        super(PNet, self).__init__()
        self.is_train = is_train

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 10, 3, 1)),
            ('prelu1', nn.PReLU(10)),
            ('pool1', nn.MaxPool2d((3,5), ceil_mode=True)),     

            ('conv2', nn.Conv2d(10, 16, (3,5), 1)),
            ('prelu2', nn.PReLU(16)),

            ('conv3', nn.Conv2d(16, 32, (3,5), 1)),
            ('prelu3', nn.PReLU(32))
        ]))

        self.conv4_1 = nn.Conv2d(32, 2, 1, 1)
        self.conv4_2 = nn.Conv2d(32, 4, 1, 1)

    def forward(self, x):

        x = self.features(x)
        a = self.conv4_1(x)
        b = self.conv4_2(x)

        if self.is_train is False:
            a = F.softmax(a, dim=1)

        return b, a

class ONet(nn.Module):

    def __init__(self, is_train=False):

        super(ONet, self).__init__()
        self.is_train = is_train

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 32, 3, 1)),
            ('prelu1', nn.PReLU(32)),
            ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv2', nn.Conv2d(32, 64, (5,3), 1)),
            ('prelu2', nn.PReLU(64)),
            ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv3', nn.Conv2d(64, 64, (5,3), 1)),
            ('prelu3', nn.PReLU(64)),
            ('pool3', nn.MaxPool2d(2, 2, ceil_mode=True)),

            ('conv4', nn.Conv2d(64, 128, 1, 1)),
            ('prelu4', nn.PReLU(128)),

            ('flatten', Flatten()),
            ('conv5', nn.Linear(1280, 256)),      #mario.c3
            ('drop5', nn.Dropout(0.25)),
            ('prelu5', nn.PReLU(256)),
        ]))

        self.conv6_1 = nn.Linear(256, 2)
        self.conv6_2 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.features(x)
        a = self.conv6_1(x)
        b = self.conv6_2(x)

        if self.is_train is False:
            a = F.softmax(a, dim=1)

        return b, a
    
class licenseBone(nn.Module):
    def __init__(self):
        super(licenseBone, self).__init__()

        layer0 = layer.Conv2dBatchReLU(1, 32, 3, 2)       #64
        layer1 = layer.Conv2dBatchReLU(32, 64, 3, 1)
        layer2 = layer.Conv2dBatchReLU(64, 64, 3, 2)      #32
        layer3 = layer.Conv2dBatchReLU(64, 128, 3, 1)
        layer4 = layer.Conv2dBatchReLU(128, 128, 3, 2)    #16

        self.layers = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
            layer4
        )

        layer00 = layer.Conv2dBatchReLU(128, 128, 3, 1)
        layer11 = layer.Conv2dBatchReLU(128, 256, 3, 2)
        layer22 = layer.Conv2dBatchReLU(256, 512, 3, 2)
        layer33 = layer.GlobalAvgPool2d()
        layer44 = layer.FullyConnectLayer(512, 8)

        self.layers0 = nn.Sequential(
            layer00,
            layer11,
            layer22,
            layer33,
            layer44
        )

    def forward(self, x):
        x0 = self.layers(x)
        x1 = self.layers0(x0)

        return [x1]

if __name__ == '__main__':

    protofile = "./landmark.prototxt"
    modelFile = "./landmark.caffemodel"

    modelPath = './models/landmark.pkl'
    #caffe.set_mode_cpu()

    net = caffe.Net(protofile, caffe.TEST)
    caffeParams = net.params

    for k in sorted(caffeParams):
        print(k)
    print(len(caffeParams))
    
    ## pnet, onet
    # network = ONet()
    # network.load_state_dict(torch.load(modelPath))
    # network.eval()

    ## landmark
    network = torch.load(modelPath)
    network.eval()

    for param_tensor, value in network.state_dict().items():
        print(param_tensor)
        # print(value.shape)
        # if "bias" in param_tensor:
        #     print(value)
    print((len(network.state_dict())))

    recycle = 0
    layerNum = 1
    i = 0
    sizeNum = 0
    # ## pnet
    # nameDict = {
    #             "features.conv1.weight":"Convolution1,0",
    #             "features.conv1.bias":"Convolution1,1",
    #             "features.prelu1.weight":"conv1,0",
    #             "features.conv2.weight":"Convolution2,0",
    #             "features.conv2.bias":"Convolution2,1",
    #             "features.prelu2.weight":"conv2,0",

    #             "features.conv3.weight":"Convolution3,0",
    #             "features.conv3.bias":"Convolution3,1",
    #             "features.prelu3.weight":"conv3,0",
    #             "conv4_1.weight":"conv4_1,0",
    #             "conv4_1.bias":"conv4_1,1",
    #             "conv4_2.weight":"conv4_2,0",
    #             "conv4_2.bias":"conv4_2,1",
    # }
    # ## onet
    # nameDict = {
    #             "features.conv1.weight":"Convolution1,0",
    #             "features.conv1.bias":"Convolution1,1",
    #             "features.prelu1.weight":"conv1,0",
    #             "features.conv2.weight":"Convolution2,0",
    #             "features.conv2.bias":"Convolution2,1",
    #             "features.prelu2.weight":"conv2,0",

    #             "features.conv3.weight":"Convolution3,0",
    #             "features.conv3.bias":"Convolution3,1",
    #             "features.prelu3.weight":"conv3,0",
    #             "features.conv4.weight":"Convolution4,0",
    #             "features.conv4.bias":"Convolution4,1",
    #             "features.prelu4.weight":"conv4,0",

    #             "features.conv5.weight":"fc1,0",
    #             "features.conv5.bias":"fc1,1",
    #             "features.prelu5.weight":"prelu1,0",

    #             "conv6_1.weight":"fc2,0",
    #             "conv6_1.bias":"fc2,1",
    #             "conv6_2.weight":"fc3,0",
    #             "conv6_2.bias":"fc3,1",
    # }
    # ## landmark
    nameDict = {
                "layers.0.layers.0.weight":"Convolution1,0",
                "layers.0.layers.1.weight":"Scale1,0",
                "layers.0.layers.1.bias":"Scale1,1",
                "layers.0.layers.1.running_mean":"BatchNorm1,0",
                "layers.0.layers.1.running_var":"BatchNorm1,1",
                "layers.0.layers.1.num_batches_tracked":"BatchNorm1,2",

                "layers.1.layers.0.weight":"Convolution2,0",
                "layers.1.layers.1.weight":"Scale2,0",
                "layers.1.layers.1.bias":"Scale2,1",
                "layers.1.layers.1.running_mean":"BatchNorm2,0",
                "layers.1.layers.1.running_var":"BatchNorm2,1",
                "layers.1.layers.1.num_batches_tracked":"BatchNorm2,2",

                "layers.2.layers.0.weight":"Convolution3,0",
                "layers.2.layers.1.weight":"Scale3,0",
                "layers.2.layers.1.bias":"Scale3,1",
                "layers.2.layers.1.running_mean":"BatchNorm3,0",
                "layers.2.layers.1.running_var":"BatchNorm3,1",
                "layers.2.layers.1.num_batches_tracked":"BatchNorm3,2",

                "layers.3.layers.0.weight":"Convolution4,0",
                "layers.3.layers.1.weight":"Scale4,0",
                "layers.3.layers.1.bias":"Scale4,1",
                "layers.3.layers.1.running_mean":"BatchNorm4,0",
                "layers.3.layers.1.running_var":"BatchNorm4,1",
                "layers.3.layers.1.num_batches_tracked":"BatchNorm4,2",

                "layers.4.layers.0.weight":"Convolution5,0",
                "layers.4.layers.1.weight":"Scale5,0",
                "layers.4.layers.1.bias":"Scale5,1",
                "layers.4.layers.1.running_mean":"BatchNorm5,0",
                "layers.4.layers.1.running_var":"BatchNorm5,1",
                "layers.4.layers.1.num_batches_tracked":"BatchNorm5,2",

                "layers0.0.layers.0.weight":"Convolution6,0",
                "layers0.0.layers.1.weight":"Scale6,0",
                "layers0.0.layers.1.bias":"Scale6,1",
                "layers0.0.layers.1.running_mean":"BatchNorm6,0",
                "layers0.0.layers.1.running_var":"BatchNorm6,1",
                "layers0.0.layers.1.num_batches_tracked":"BatchNorm6,2",

                "layers0.1.layers.0.weight":"Convolution7,0",
                "layers0.1.layers.1.weight":"Scale7,0",
                "layers0.1.layers.1.bias":"Scale7,1",
                "layers0.1.layers.1.running_mean":"BatchNorm7,0",
                "layers0.1.layers.1.running_var":"BatchNorm7,1",
                "layers0.1.layers.1.num_batches_tracked":"BatchNorm7,2",

                "layers0.2.layers.0.weight":"Convolution8,0",
                "layers0.2.layers.1.weight":"Scale8,0",
                "layers0.2.layers.1.bias":"Scale8,1",
                "layers0.2.layers.1.running_mean":"BatchNorm8,0",
                "layers0.2.layers.1.running_var":"BatchNorm8,1",
                "layers0.2.layers.1.num_batches_tracked":"BatchNorm8,2",

                "layers0.4.layers.0.weight":"fc1,0",
                "layers0.4.layers.0.bias":"fc1,1",
    }

    pytorchLayerNameList = list(nameDict.keys())
    caffeLayerNameList = list(nameDict.values())

    i = 0
    # check if all parameters in nameDict
    for param_tensor in network.state_dict():
        print(str(i)+' |', param_tensor+' |', nameDict[param_tensor])
        if param_tensor not in pytorchLayerNameList:
            print("there is some problem in nameDict")
            sys.exit()

        param = network.state_dict()[param_tensor]

        caffeLayerPara = nameDict[param_tensor]

        if "," in caffeLayerPara:
            caffeLayerName, caffeLayerMatNum = caffeLayerPara.strip().split(",")
            caffeLayerMatNum = int(caffeLayerMatNum)
            if caffeLayerName not in caffeParams:
                print("caffeLayerName is not in caffe")
           
            if "num_batches_tracked" in param_tensor:
                caffeParams[caffeLayerName][caffeLayerMatNum].data[...] = np.array([1.0])
            else:
                caffeParams[caffeLayerName][caffeLayerMatNum].data[...] = param.cpu().data.numpy()

        i += 1

    net.save(modelFile) 
    print("net save end")
    sys.exit()
