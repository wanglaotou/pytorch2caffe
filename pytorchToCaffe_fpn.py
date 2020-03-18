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
import caffe
from ssd_mobilenetv2_fpn import build_ssd
# import layer

print(caffe.__path__)

if __name__ == '__main__':

    protofile = "./ssd_mobilenetv2_fpn_ok.prototxt"     
    ## ssd_mobilenetv2_fpn_ok.prototxt 转换pytorch模型为caffe模型使用的prototxt
    modelFile = "./ssd_mobilenetv2_addmiss_0302.caffemodel"

    modelPath = './pytorchToCaffe/mobilenetv2_290000_addmiss.pth'
    caffe.set_mode_cpu()

    
    # step 1: transform the pytorch model weights to caffe model weights
    net = caffe.Net(protofile, caffe.TEST)
    caffeParams = net.params

    for k in sorted(caffeParams):
        print(k)
    print(len(caffeParams))
    
    # ssd
    network = build_ssd('test')
    network.load_state_dict(torch.load(modelPath))
    network.eval()

    # # landmark
    # # network = backbone.licenseBone()
    # network = torch.load(modelPath)
    # network.eval()

    # for param_tensor in network.state_dict():
    #     print(param_tensor)
    for param_tensor, value in network.state_dict().items():
        print(param_tensor)

    print((len(network.state_dict())))
   
    i = 0
    sizeNum = 0
    
    ## ssd mobilenetv2
    nameDict = {
                "mobilenet.conv1.weight":"Convolution1,0",
                "mobilenet.bn1.weight":"Scale1,0",
                "mobilenet.bn1.bias":"Scale1,1",
                "mobilenet.bn1.running_mean":"BatchNorm1,0",
                "mobilenet.bn1.running_var":"BatchNorm1,1",
                "mobilenet.bn1.num_batches_tracked":"BatchNorm1,2",
                "mobilenet.bottlenecks.Bottlenecks_0.LinearBottleneck0_0.conv1.weight":"Convolution2,0",
                "mobilenet.bottlenecks.Bottlenecks_0.LinearBottleneck0_0.bn1.weight":"Scale2,0",
                "mobilenet.bottlenecks.Bottlenecks_0.LinearBottleneck0_0.bn1.bias":"Scale2,1",
                "mobilenet.bottlenecks.Bottlenecks_0.LinearBottleneck0_0.bn1.running_mean":"BatchNorm2,0",
                "mobilenet.bottlenecks.Bottlenecks_0.LinearBottleneck0_0.bn1.running_var":"BatchNorm2,1",
                "mobilenet.bottlenecks.Bottlenecks_0.LinearBottleneck0_0.bn1.num_batches_tracked":"BatchNorm2,2",
                "mobilenet.bottlenecks.Bottlenecks_0.LinearBottleneck0_0.conv2.weight":"Convolution3,0",
                "mobilenet.bottlenecks.Bottlenecks_0.LinearBottleneck0_0.bn2.weight":"Scale3,0",
                "mobilenet.bottlenecks.Bottlenecks_0.LinearBottleneck0_0.bn2.bias":"Scale3,1",
                "mobilenet.bottlenecks.Bottlenecks_0.LinearBottleneck0_0.bn2.running_mean":"BatchNorm3,0",
                "mobilenet.bottlenecks.Bottlenecks_0.LinearBottleneck0_0.bn2.running_var":"BatchNorm3,1",
                "mobilenet.bottlenecks.Bottlenecks_0.LinearBottleneck0_0.bn2.num_batches_tracked":"BatchNorm3,2",
                "mobilenet.bottlenecks.Bottlenecks_0.LinearBottleneck0_0.conv3.weight":"Convolution4,0",
                "mobilenet.bottlenecks.Bottlenecks_0.LinearBottleneck0_0.bn3.weight":"conv3,0",
                "mobilenet.bottlenecks.Bottlenecks_0.LinearBottleneck0_0.bn3.bias":"conv3,1",
                "mobilenet.bottlenecks.Bottlenecks_0.LinearBottleneck0_0.bn3.running_mean":"BatchNorm4,0",
                "mobilenet.bottlenecks.Bottlenecks_0.LinearBottleneck0_0.bn3.running_var":"BatchNorm4,1",
                "mobilenet.bottlenecks.Bottlenecks_0.LinearBottleneck0_0.bn3.num_batches_tracked":"BatchNorm4,2",
                "mobilenet.bottlenecks.Bottlenecks_1.LinearBottleneck1_0.conv1.weight":"Convolution5,0",
                "mobilenet.bottlenecks.Bottlenecks_1.LinearBottleneck1_0.bn1.weight":"Scale4,0",
                "mobilenet.bottlenecks.Bottlenecks_1.LinearBottleneck1_0.bn1.bias":"Scale4,1",
                "mobilenet.bottlenecks.Bottlenecks_1.LinearBottleneck1_0.bn1.running_mean":"BatchNorm5,0",
                "mobilenet.bottlenecks.Bottlenecks_1.LinearBottleneck1_0.bn1.running_var":"BatchNorm5,1",
                "mobilenet.bottlenecks.Bottlenecks_1.LinearBottleneck1_0.bn1.num_batches_tracked":"BatchNorm5,2",
                "mobilenet.bottlenecks.Bottlenecks_1.LinearBottleneck1_0.conv2.weight":"Convolution6,0",
                "mobilenet.bottlenecks.Bottlenecks_1.LinearBottleneck1_0.bn2.weight":"Scale5,0",
                "mobilenet.bottlenecks.Bottlenecks_1.LinearBottleneck1_0.bn2.bias":"Scale5,1",
                "mobilenet.bottlenecks.Bottlenecks_1.LinearBottleneck1_0.bn2.running_mean":"BatchNorm6,0",
                "mobilenet.bottlenecks.Bottlenecks_1.LinearBottleneck1_0.bn2.running_var":"BatchNorm6,1",
                "mobilenet.bottlenecks.Bottlenecks_1.LinearBottleneck1_0.bn2.num_batches_tracked":"BatchNorm6,2",
                "mobilenet.bottlenecks.Bottlenecks_1.LinearBottleneck1_0.conv3.weight":"Convolution7,0",
                "mobilenet.bottlenecks.Bottlenecks_1.LinearBottleneck1_0.bn3.weight":"conv6,0",
                "mobilenet.bottlenecks.Bottlenecks_1.LinearBottleneck1_0.bn3.bias":"conv6,1",
                "mobilenet.bottlenecks.Bottlenecks_1.LinearBottleneck1_0.bn3.running_mean":"BatchNorm7,0",
                "mobilenet.bottlenecks.Bottlenecks_1.LinearBottleneck1_0.bn3.running_var":"BatchNorm7,1",
                "mobilenet.bottlenecks.Bottlenecks_1.LinearBottleneck1_0.bn3.num_batches_tracked":"BatchNorm7,2",
                "mobilenet.bottlenecks.Bottlenecks_1.LinearBottleneck1_1.conv1.weight":"Convolution8,0",
                "mobilenet.bottlenecks.Bottlenecks_1.LinearBottleneck1_1.bn1.weight":"Scale6,0",
                "mobilenet.bottlenecks.Bottlenecks_1.LinearBottleneck1_1.bn1.bias":"Scale6,1",
                "mobilenet.bottlenecks.Bottlenecks_1.LinearBottleneck1_1.bn1.running_mean":"BatchNorm8,0",
                "mobilenet.bottlenecks.Bottlenecks_1.LinearBottleneck1_1.bn1.running_var":"BatchNorm8,1",
                "mobilenet.bottlenecks.Bottlenecks_1.LinearBottleneck1_1.bn1.num_batches_tracked":"BatchNorm8,2",
                "mobilenet.bottlenecks.Bottlenecks_1.LinearBottleneck1_1.conv2.weight":"Convolution9,0",
                "mobilenet.bottlenecks.Bottlenecks_1.LinearBottleneck1_1.bn2.weight":"Scale7,0",
                "mobilenet.bottlenecks.Bottlenecks_1.LinearBottleneck1_1.bn2.bias":"Scale7,1",
                "mobilenet.bottlenecks.Bottlenecks_1.LinearBottleneck1_1.bn2.running_mean":"BatchNorm9,0",
                "mobilenet.bottlenecks.Bottlenecks_1.LinearBottleneck1_1.bn2.running_var":"BatchNorm9,1",
                "mobilenet.bottlenecks.Bottlenecks_1.LinearBottleneck1_1.bn2.num_batches_tracked":"BatchNorm9,2",
                "mobilenet.bottlenecks.Bottlenecks_1.LinearBottleneck1_1.conv3.weight":"Convolution10,0",
                "mobilenet.bottlenecks.Bottlenecks_1.LinearBottleneck1_1.bn3.weight":"conv9,0",
                "mobilenet.bottlenecks.Bottlenecks_1.LinearBottleneck1_1.bn3.bias":"conv9,1",
                "mobilenet.bottlenecks.Bottlenecks_1.LinearBottleneck1_1.bn3.running_mean":"BatchNorm10,0",
                "mobilenet.bottlenecks.Bottlenecks_1.LinearBottleneck1_1.bn3.running_var":"BatchNorm10,1",
                "mobilenet.bottlenecks.Bottlenecks_1.LinearBottleneck1_1.bn3.num_batches_tracked":"BatchNorm10,2",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_0.conv1.weight":"Convolution11,0",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_0.bn1.weight":"Scale8,0",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_0.bn1.bias":"Scale8,1",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_0.bn1.running_mean":"BatchNorm11,0",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_0.bn1.running_var":"BatchNorm11,1",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_0.bn1.num_batches_tracked":"BatchNorm11,2",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_0.conv2.weight":"Convolution12,0",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_0.bn2.weight":"Scale9,0",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_0.bn2.bias":"Scale9,1",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_0.bn2.running_mean":"BatchNorm12,0",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_0.bn2.running_var":"BatchNorm12,1",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_0.bn2.num_batches_tracked":"BatchNorm12,2",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_0.conv3.weight":"Convolution13,0",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_0.bn3.weight":"conv12,0",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_0.bn3.bias":"conv12,1",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_0.bn3.running_mean":"BatchNorm13,0",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_0.bn3.running_var":"BatchNorm13,1",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_0.bn3.num_batches_tracked":"BatchNorm13,2",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_1.conv1.weight":"Convolution14,0",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_1.bn1.weight":"Scale10,0",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_1.bn1.bias":"Scale10,1",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_1.bn1.running_mean":"BatchNorm14,0",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_1.bn1.running_var":"BatchNorm14,1",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_1.bn1.num_batches_tracked":"BatchNorm14,2",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_1.conv2.weight":"Convolution15,0",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_1.bn2.weight":"Scale11,0",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_1.bn2.bias":"Scale11,1",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_1.bn2.running_mean":"BatchNorm15,0",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_1.bn2.running_var":"BatchNorm15,1",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_1.bn2.num_batches_tracked":"BatchNorm15,2",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_1.conv3.weight":"Convolution16,0",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_1.bn3.weight":"conv15,0",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_1.bn3.bias":"conv15,1",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_1.bn3.running_mean":"BatchNorm16,0",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_1.bn3.running_var":"BatchNorm16,1",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_1.bn3.num_batches_tracked":"BatchNorm16,2",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_2.conv1.weight":"Convolution17,0",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_2.bn1.weight":"Scale12,0",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_2.bn1.bias":"Scale12,1",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_2.bn1.running_mean":"BatchNorm17,0",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_2.bn1.running_var":"BatchNorm17,1",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_2.bn1.num_batches_tracked":"BatchNorm17,2",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_2.conv2.weight":"Convolution18,0",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_2.bn2.weight":"Scale13,0",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_2.bn2.bias":"Scale13,1",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_2.bn2.running_mean":"BatchNorm18,0",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_2.bn2.running_var":"BatchNorm18,1",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_2.bn2.num_batches_tracked":"BatchNorm18,2",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_2.conv3.weight":"Convolution19,0",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_2.bn3.weight":"conv18,0",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_2.bn3.bias":"conv18,1",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_2.bn3.running_mean":"BatchNorm19,0",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_2.bn3.running_var":"BatchNorm19,1",
                "mobilenet.bottlenecks.Bottlenecks_2.LinearBottleneck2_2.bn3.num_batches_tracked":"BatchNorm19,2",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_0.conv1.weight":"Convolution20,0",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_0.bn1.weight":"Scale14,0",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_0.bn1.bias":"Scale14,1",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_0.bn1.running_mean":"BatchNorm20,0",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_0.bn1.running_var":"BatchNorm20,1",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_0.bn1.num_batches_tracked":"BatchNorm20,2",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_0.conv2.weight":"Convolution21,0",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_0.bn2.weight":"Scale15,0",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_0.bn2.bias":"Scale15,1",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_0.bn2.running_mean":"BatchNorm21,0",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_0.bn2.running_var":"BatchNorm21,1",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_0.bn2.num_batches_tracked":"BatchNorm21,2",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_0.conv3.weight":"Convolution22,0",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_0.bn3.weight":"conv21,0",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_0.bn3.bias":"conv21,1",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_0.bn3.running_mean":"BatchNorm22,0",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_0.bn3.running_var":"BatchNorm22,1",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_0.bn3.num_batches_tracked":"BatchNorm22,2",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_1.conv1.weight":"Convolution23,0",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_1.bn1.weight":"Scale16,0",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_1.bn1.bias":"Scale16,1",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_1.bn1.running_mean":"BatchNorm23,0",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_1.bn1.running_var":"BatchNorm23,1",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_1.bn1.num_batches_tracked":"BatchNorm23,2",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_1.conv2.weight":"Convolution24,0",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_1.bn2.weight":"Scale17,0",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_1.bn2.bias":"Scale17,1",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_1.bn2.running_mean":"BatchNorm24,0",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_1.bn2.running_var":"BatchNorm24,1",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_1.bn2.num_batches_tracked":"BatchNorm24,2",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_1.conv3.weight":"Convolution25,0",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_1.bn3.weight":"conv24,0",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_1.bn3.bias":"conv24,1",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_1.bn3.running_mean":"BatchNorm25,0",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_1.bn3.running_var":"BatchNorm25,1",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_1.bn3.num_batches_tracked":"BatchNorm25,2",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_2.conv1.weight":"Convolution26,0",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_2.bn1.weight":"Scale18,0",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_2.bn1.bias":"Scale18,1",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_2.bn1.running_mean":"BatchNorm26,0",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_2.bn1.running_var":"BatchNorm26,1",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_2.bn1.num_batches_tracked":"BatchNorm26,2",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_2.conv2.weight":"Convolution27,0",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_2.bn2.weight":"Scale19,0",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_2.bn2.bias":"Scale19,1",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_2.bn2.running_mean":"BatchNorm27,0",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_2.bn2.running_var":"BatchNorm27,1",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_2.bn2.num_batches_tracked":"BatchNorm27,2",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_2.conv3.weight":"Convolution28,0",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_2.bn3.weight":"conv27,0",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_2.bn3.bias":"conv27,1",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_2.bn3.running_mean":"BatchNorm28,0",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_2.bn3.running_var":"BatchNorm28,1",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_2.bn3.num_batches_tracked":"BatchNorm28,2",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_3.conv1.weight":"Convolution29,0",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_3.bn1.weight":"Scale20,0",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_3.bn1.bias":"Scale20,1",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_3.bn1.running_mean":"BatchNorm29,0",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_3.bn1.running_var":"BatchNorm29,1",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_3.bn1.num_batches_tracked":"BatchNorm29,2",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_3.conv2.weight":"Convolution30,0",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_3.bn2.weight":"Scale21,0",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_3.bn2.bias":"Scale21,1",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_3.bn2.running_mean":"BatchNorm30,0",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_3.bn2.running_var":"BatchNorm30,1",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_3.bn2.num_batches_tracked":"BatchNorm30,2",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_3.conv3.weight":"Convolution31,0",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_3.bn3.weight":"conv30,0",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_3.bn3.bias":"conv30,1",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_3.bn3.running_mean":"BatchNorm31,0",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_3.bn3.running_var":"BatchNorm31,1",
                "mobilenet.bottlenecks.Bottlenecks_3.LinearBottleneck3_3.bn3.num_batches_tracked":"BatchNorm31,2",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_0.conv1.weight":"Convolution32,0",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_0.bn1.weight":"Scale22,0",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_0.bn1.bias":"Scale22,1",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_0.bn1.running_mean":"BatchNorm32,0",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_0.bn1.running_var":"BatchNorm32,1",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_0.bn1.num_batches_tracked":"BatchNorm32,2",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_0.conv2.weight":"Convolution33,0",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_0.bn2.weight":"Scale23,0",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_0.bn2.bias":"Scale23,1",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_0.bn2.running_mean":"BatchNorm33,0",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_0.bn2.running_var":"BatchNorm33,1",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_0.bn2.num_batches_tracked":"BatchNorm33,2",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_0.conv3.weight":"Convolution34,0",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_0.bn3.weight":"conv33,0",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_0.bn3.bias":"conv33,1",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_0.bn3.running_mean":"BatchNorm34,0",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_0.bn3.running_var":"BatchNorm34,1",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_0.bn3.num_batches_tracked":"BatchNorm34,2",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_1.conv1.weight":"Convolution35,0",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_1.bn1.weight":"Scale24,0",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_1.bn1.bias":"Scale24,1",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_1.bn1.running_mean":"BatchNorm35,0",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_1.bn1.running_var":"BatchNorm35,1",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_1.bn1.num_batches_tracked":"BatchNorm35,2",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_1.conv2.weight":"Convolution36,0",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_1.bn2.weight":"Scale25,0",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_1.bn2.bias":"Scale25,1",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_1.bn2.running_mean":"BatchNorm36,0",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_1.bn2.running_var":"BatchNorm36,1",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_1.bn2.num_batches_tracked":"BatchNorm36,2",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_1.conv3.weight":"Convolution37,0",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_1.bn3.weight":"conv36,0",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_1.bn3.bias":"conv36,1",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_1.bn3.running_mean":"BatchNorm37,0",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_1.bn3.running_var":"BatchNorm37,1",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_1.bn3.num_batches_tracked":"BatchNorm37,2",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_2.conv1.weight":"Convolution38,0",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_2.bn1.weight":"Scale26,0",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_2.bn1.bias":"Scale26,1",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_2.bn1.running_mean":"BatchNorm38,0",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_2.bn1.running_var":"BatchNorm38,1",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_2.bn1.num_batches_tracked":"BatchNorm38,2",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_2.conv2.weight":"Convolution39,0",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_2.bn2.weight":"Scale27,0",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_2.bn2.bias":"Scale27,1",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_2.bn2.running_mean":"BatchNorm39,0",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_2.bn2.running_var":"BatchNorm39,1",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_2.bn2.num_batches_tracked":"BatchNorm39,2",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_2.conv3.weight":"Convolution40,0",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_2.bn3.weight":"conv39,0",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_2.bn3.bias":"conv39,1",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_2.bn3.running_mean":"BatchNorm40,0",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_2.bn3.running_var":"BatchNorm40,1",
                "mobilenet.bottlenecks.Bottlenecks_4.LinearBottleneck4_2.bn3.num_batches_tracked":"BatchNorm40,2",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_0.conv1.weight":"Convolution41,0",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_0.bn1.weight":"Scale28,0",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_0.bn1.bias":"Scale28,1",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_0.bn1.running_mean":"BatchNorm41,0",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_0.bn1.running_var":"BatchNorm41,1",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_0.bn1.num_batches_tracked":"BatchNorm41,2",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_0.conv2.weight":"Convolution42,0",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_0.bn2.weight":"Scale29,0",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_0.bn2.bias":"Scale29,1",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_0.bn2.running_mean":"BatchNorm42,0",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_0.bn2.running_var":"BatchNorm42,1",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_0.bn2.num_batches_tracked":"BatchNorm42,2",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_0.conv3.weight":"Convolution43,0",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_0.bn3.weight":"conv42,0",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_0.bn3.bias":"conv42,1",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_0.bn3.running_mean":"BatchNorm43,0",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_0.bn3.running_var":"BatchNorm43,1",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_0.bn3.num_batches_tracked":"BatchNorm43,2",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_1.conv1.weight":"Convolution44,0",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_1.bn1.weight":"Scale30,0",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_1.bn1.bias":"Scale30,1",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_1.bn1.running_mean":"BatchNorm44,0",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_1.bn1.running_var":"BatchNorm44,1",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_1.bn1.num_batches_tracked":"BatchNorm44,2",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_1.conv2.weight":"Convolution45,0",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_1.bn2.weight":"Scale31,0",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_1.bn2.bias":"Scale31,1",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_1.bn2.running_mean":"BatchNorm45,0",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_1.bn2.running_var":"BatchNorm45,1",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_1.bn2.num_batches_tracked":"BatchNorm45,2",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_1.conv3.weight":"Convolution46,0",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_1.bn3.weight":"conv45,0",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_1.bn3.bias":"conv45,1",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_1.bn3.running_mean":"BatchNorm46,0",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_1.bn3.running_var":"BatchNorm46,1",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_1.bn3.num_batches_tracked":"BatchNorm46,2",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_2.conv1.weight":"Convolution47,0",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_2.bn1.weight":"Scale32,0",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_2.bn1.bias":"Scale32,1",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_2.bn1.running_mean":"BatchNorm47,0",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_2.bn1.running_var":"BatchNorm47,1",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_2.bn1.num_batches_tracked":"BatchNorm47,2",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_2.conv2.weight":"Convolution48,0",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_2.bn2.weight":"Scale33,0",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_2.bn2.bias":"Scale33,1",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_2.bn2.running_mean":"BatchNorm48,0",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_2.bn2.running_var":"BatchNorm48,1",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_2.bn2.num_batches_tracked":"BatchNorm48,2",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_2.conv3.weight":"Convolution49,0",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_2.bn3.weight":"conv48,0",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_2.bn3.bias":"conv48,1",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_2.bn3.running_mean":"BatchNorm49,0",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_2.bn3.running_var":"BatchNorm49,1",
                "mobilenet.bottlenecks.Bottlenecks_5.LinearBottleneck5_2.bn3.num_batches_tracked":"BatchNorm49,2",
                "mobilenet.bottlenecks.Bottlenecks_6.LinearBottleneck6_0.conv1.weight":"Convolution50,0",
                "mobilenet.bottlenecks.Bottlenecks_6.LinearBottleneck6_0.bn1.weight":"Scale34,0",
                "mobilenet.bottlenecks.Bottlenecks_6.LinearBottleneck6_0.bn1.bias":"Scale34,1",
                "mobilenet.bottlenecks.Bottlenecks_6.LinearBottleneck6_0.bn1.running_mean":"BatchNorm50,0",
                "mobilenet.bottlenecks.Bottlenecks_6.LinearBottleneck6_0.bn1.running_var":"BatchNorm50,1",
                "mobilenet.bottlenecks.Bottlenecks_6.LinearBottleneck6_0.bn1.num_batches_tracked":"BatchNorm50,2",
                "mobilenet.bottlenecks.Bottlenecks_6.LinearBottleneck6_0.conv2.weight":"Convolution51,0",
                "mobilenet.bottlenecks.Bottlenecks_6.LinearBottleneck6_0.bn2.weight":"Scale35,0",
                "mobilenet.bottlenecks.Bottlenecks_6.LinearBottleneck6_0.bn2.bias":"Scale35,1",
                "mobilenet.bottlenecks.Bottlenecks_6.LinearBottleneck6_0.bn2.running_mean":"BatchNorm51,0",
                "mobilenet.bottlenecks.Bottlenecks_6.LinearBottleneck6_0.bn2.running_var":"BatchNorm51,1",
                "mobilenet.bottlenecks.Bottlenecks_6.LinearBottleneck6_0.bn2.num_batches_tracked":"BatchNorm51,2",
                "mobilenet.bottlenecks.Bottlenecks_6.LinearBottleneck6_0.conv3.weight":"Convolution52,0",
                "mobilenet.bottlenecks.Bottlenecks_6.LinearBottleneck6_0.bn3.weight":"conv51,0",
                "mobilenet.bottlenecks.Bottlenecks_6.LinearBottleneck6_0.bn3.bias":"conv51,1",
                "mobilenet.bottlenecks.Bottlenecks_6.LinearBottleneck6_0.bn3.running_mean":"BatchNorm52,0",
                "mobilenet.bottlenecks.Bottlenecks_6.LinearBottleneck6_0.bn3.running_var":"BatchNorm52,1",
                "mobilenet.bottlenecks.Bottlenecks_6.LinearBottleneck6_0.bn3.num_batches_tracked":"BatchNorm52,2",
                "mobilenet.conv_last.weight":"Convolution57,0",
                "mobilenet.bn_last.weight":"Scale36,0",
                "mobilenet.bn_last.bias":"Scale36,1",
                "mobilenet.bn_last.running_mean":"BatchNorm53,0",
                "mobilenet.bn_last.running_var":"BatchNorm53,1",
                "mobilenet.bn_last.num_batches_tracked":"BatchNorm53,2",
                ## copy weights needs, but check weights no need.
                "mobilenet.fc.weight":"",       ## no weigths, but copy weights needs
                "mobilenet.fc.bias":"",       
                "L2Norm.weight":"",
                "extras.0.0.weight":"Convolution58,0",
                "extras.0.1.weight":"Scale37,0",
                "extras.0.1.bias":"Scale37,1",
                "extras.0.1.running_mean":"BatchNorm54,0",
                "extras.0.1.running_var":"BatchNorm54,1",
                "extras.0.1.num_batches_tracked":"BatchNorm54,2",
                "extras.1.0.weight":"Convolution59,0",
                "extras.1.1.weight":"Scale38,0",
                "extras.1.1.bias":"Scale38,1",
                "extras.1.1.running_mean":"BatchNorm55,0",
                "extras.1.1.running_var":"BatchNorm55,1",
                "extras.1.1.num_batches_tracked":"BatchNorm55,2",
                "extras.2.0.weight":"Convolution60,0",
                "extras.2.1.weight":"Scale39,0",
                "extras.2.1.bias":"Scale39,1",
                "extras.2.1.running_mean":"BatchNorm56,0",
                "extras.2.1.running_var":"BatchNorm56,1",
                "extras.2.1.num_batches_tracked":"BatchNorm56,2",
                "extras.3.0.weight":"Convolution61,0",
                "extras.3.1.weight":"Scale40,0",
                "extras.3.1.bias":"Scale40,1",
                "extras.3.1.running_mean":"BatchNorm57,0",
                "extras.3.1.running_var":"BatchNorm57,1",
                "extras.3.1.num_batches_tracked":"BatchNorm57,2",
                "extras.4.0.weight":"Convolution62,0",
                "extras.4.1.weight":"Scale41,0",
                "extras.4.1.bias":"Scale41,1",
                "extras.4.1.running_mean":"BatchNorm58,0",
                "extras.4.1.running_var":"BatchNorm58,1",
                "extras.4.1.num_batches_tracked":"BatchNorm58,2",
                "extras.5.0.weight":"Convolution63,0",
                "extras.5.1.weight":"Scale42,0",
                "extras.5.1.bias":"Scale42,1",
                "extras.5.1.running_mean":"BatchNorm59,0",
                "extras.5.1.running_var":"BatchNorm59,1",
                "extras.5.1.num_batches_tracked":"BatchNorm59,2",
                "extras.6.0.weight":"Convolution64,0",
                "extras.6.1.weight":"Scale43,0",
                "extras.6.1.bias":"Scale43,1",
                "extras.6.1.running_mean":"BatchNorm60,0",
                "extras.6.1.running_var":"BatchNorm60,1",
                "extras.6.1.num_batches_tracked":"BatchNorm60,2",
                "extras.7.0.weight":"Convolution65,0",
                "extras.7.1.weight":"Scale44,0",
                "extras.7.1.bias":"Scale44,1",
                "extras.7.1.running_mean":"BatchNorm61,0",
                "extras.7.1.running_var":"BatchNorm61,1",
                "extras.7.1.num_batches_tracked":"BatchNorm61,2",
                "fpn.P5_1.weight":"Convolution53,0",
                "fpn.P5_1.bias":"Convolution53,1",
                "fpn.P5_upsampled.weight":"Deconvolution1,0",
                "fpn.P5_upsampled.bias":"Deconvolution1,1",
                "fpn.P5_2.weight":"Convolution54,0",
                "fpn.P5_2.bias":"Convolution54,1",
                "fpn.P4_1.weight":"Convolution55,0",
                "fpn.P4_1.bias":"Convolution55,1",
                ## copy weights needs, but check weights no need.
                "fpn.P4_upsampled.weight":"",     ## no weigths, but copy weights needs
                "fpn.P4_upsampled.bias":"",
                "fpn.P4_2.weight":"Convolution56,0",
                "fpn.P4_2.bias":"Convolution56,1",
                
                "loc.0.weight":"loc1,0",
                "loc.0.bias":"loc1,1",
                "loc.1.weight":"loc2,0",
                "loc.1.bias":"loc2,1",
                "loc.2.weight":"loc3,0",
                "loc.2.bias":"loc3,1",
                "loc.3.weight":"loc4,0",
                "loc.3.bias":"loc4,1",
                "loc.4.weight":"loc5,0",
                "loc.4.bias":"loc5,1",
                "loc.5.weight":"loc6,0",
                "loc.5.bias":"loc6,1",
                "conf.0.weight":"conf1,0",
                "conf.0.bias":"conf1,1",
                "conf.1.weight":"conf2,0",
                "conf.1.bias":"conf2,1",
                "conf.2.weight":"conf3,0",
                "conf.2.bias":"conf3,1",
                "conf.3.weight":"conf4,0",
                "conf.3.bias":"conf4,1",
                "conf.4.weight":"conf5,0",
                "conf.4.bias":"conf5,1",
                "conf.5.weight":"conf6,0",
                "conf.5.bias":"conf6,1",

    }
    
    # ## landmark
    # nameDict = {
    #     "layers.0.layers.0.weight":"Convolution1,0",
    #     "layers.0.layers.1.weight":"Scale1,0",
    #     "layers.0.layers.1.bias":"Scale1,1",
    #     "layers.0.layers.1.running_mean":"BatchNorm1,0",
    #     "layers.0.layers.1.running_var":"BatchNorm1,1",
    #     "layers.0.layers.1.num_batches_tracked":"BatchNorm1,2",

    #     "layers.1.layers.0.weight":"Convolution2,0",
    #     "layers.1.layers.1.weight":"Scale2,0",
    #     "layers.1.layers.1.bias":"Scale2,1",
    #     "layers.1.layers.1.running_mean":"BatchNorm2,0",
    #     "layers.1.layers.1.running_var":"BatchNorm2,1",
    #     "layers.1.layers.1.num_batches_tracked":"BatchNorm2,2",

    #     "layers.2.layers.0.weight":"Convolution3,0",
    #     "layers.2.layers.1.weight":"Scale3,0",
    #     "layers.2.layers.1.bias":"Scale3,1",
    #     "layers.2.layers.1.running_mean":"BatchNorm3,0",
    #     "layers.2.layers.1.running_var":"BatchNorm3,1",
    #     "layers.2.layers.1.num_batches_tracked":"BatchNorm3,2",

    #     "layers.3.layers.0.weight":"Convolution4,0",
    #     "layers.3.layers.1.weight":"Scale4,0",
    #     "layers.3.layers.1.bias":"Scale4,1",
    #     "layers.3.layers.1.running_mean":"BatchNorm4,0",
    #     "layers.3.layers.1.running_var":"BatchNorm4,1",
    #     "layers.3.layers.1.num_batches_tracked":"BatchNorm4,2",

    #     "layers.4.layers.0.weight":"Convolution5,0",
    #     "layers.4.layers.1.weight":"Scale5,0",
    #     "layers.4.layers.1.bias":"Scale5,1",
    #     "layers.4.layers.1.running_mean":"BatchNorm5,0",
    #     "layers.4.layers.1.running_var":"BatchNorm5,1",
    #     "layers.4.layers.1.num_batches_tracked":"BatchNorm5,2",

    #     "layers0.0.layers.0.weight":"Convolution6,0",
    #     "layers0.0.layers.1.weight":"Scale6,0",
    #     "layers0.0.layers.1.bias":"Scale6,1",
    #     "layers0.0.layers.1.running_mean":"BatchNorm6,0",
    #     "layers0.0.layers.1.running_var":"BatchNorm6,1",
    #     "layers0.0.layers.1.num_batches_tracked":"BatchNorm6,2",

    #     "layers0.1.layers.0.weight":"Convolution7,0",
    #     "layers0.1.layers.1.weight":"Scale7,0",
    #     "layers0.1.layers.1.bias":"Scale7,1",
    #     "layers0.1.layers.1.running_mean":"BatchNorm7,0",
    #     "layers0.1.layers.1.running_var":"BatchNorm7,1",
    #     "layers0.1.layers.1.num_batches_tracked":"BatchNorm7,2",

    #     "layers0.2.layers.0.weight":"Convolution8,0",
    #     "layers0.2.layers.1.weight":"Scale8,0",
    #     "layers0.2.layers.1.bias":"Scale8,0",
    #     "layers0.2.layers.1.running_mean":"BatchNorm8,0",
    #     "layers0.2.layers.1.running_var":"BatchNorm8,1",
    #     "layers0.2.layers.1.num_batches_tracked":"BatchNorm8,2",

    #     "layers0.4.layers.0.weight":"fc22,0",
    #     "layers0.4.layers.0.bias":"fc22,1",
    # }

    
    ## step 1: 将pytorch模型的参数转换为caffe模型参数
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
    
    # i = 0
    # # check if all parameters in nameDict
    # for param_tensor in network.state_dict():
    #     print(str(i)+' |', param_tensor+' |', nameDict[param_tensor])

    #     if param_tensor not in pytorchLayerNameList:
    #         print("there is some problem in nameDict")
    #         sys.exit()
    #     else:
    #         param = network.state_dict()[param_tensor]
    #         # print('param.cpu().data.numpy():', param.cpu().data.numpy().shape)
    #         print('param_tensor:', param_tensor)
    #         caffeLayerPara = nameDict[param_tensor]
    #         # print('caffeLayerPara:', caffeLayerPara)

    #         if "," in caffeLayerPara:
    #             caffeLayerName, caffeLayerMatNum = caffeLayerPara.strip().split(",")
    #             caffeLayerMatNum = int(caffeLayerMatNum)
    #             if caffeLayerName not in caffeParams:
    #                 print("caffeLayerName is not in caffe")
    #             print('param.cpu().data.numpy():', param.cpu().data.numpy().shape)
    #             # print(caffeParams[caffeLayerName][caffeLayerMatNum].data[...])
    #             print('caffe layer shape:', caffeParams[caffeLayerName][caffeLayerMatNum].data[...].shape)
    #             print('==================================')
    #             if "num_batches_tracked" in param_tensor:
    #                 caffeParams[caffeLayerName][caffeLayerMatNum].data[...] = np.array([1.0])
    #             else:
    #                 caffeParams[caffeLayerName][caffeLayerMatNum].data[...] = param.cpu().data.numpy()

    #         i += 1

        ## 查错
        # if param_tensor in pytorchLayerNameList:
        #     print("there is some problem in nameDict")
        #     # sys.exit()

        #     param = network.state_dict()[param_tensor]
        #     # print('param.cpu().data.numpy():', param.cpu().data.numpy().shape)
        #     print('param_tensor:', param_tensor)
        #     caffeLayerPara = nameDict[param_tensor]
        #     # print('caffeLayerPara:', caffeLayerPara)

        #     if "," in caffeLayerPara:
        #         caffeLayerName, caffeLayerMatNum = caffeLayerPara.strip().split(",")
        #         caffeLayerMatNum = int(caffeLayerMatNum)
        #         if caffeLayerName not in caffeParams:
        #             print("caffeLayerName is not in caffe")
        #             break
        #         print('param.cpu().data.numpy():', param.cpu().data.numpy().shape)
        #         # print(caffeParams[caffeLayerName][caffeLayerMatNum].data[...])
        #         print('caffe layer shape:', caffeParams[caffeLayerName][caffeLayerMatNum].data[...].shape)
        #         print('==================================')
        #         if "num_batches_tracked" in param_tensor:
        #             caffeParams[caffeLayerName][caffeLayerMatNum].data[...] = np.array([1.0])
        #         else:
        #             caffeParams[caffeLayerName][caffeLayerMatNum].data[...] = param.cpu().data.numpy()

    net.save(modelFile) 
    print("net save end")
    # sys.exit()
    
    
    '''
    ## this part is no need, just for further debug
    ## check the pytorch model weights and caffe model weights
    # caffenet = caffe.Net(protofile, modelFile, caffe.TEST)

    # for k,v in caffe_net.params.items():
    #     print(k,v[0].data.shape)
    # print('===============================================')
    # # for k,v in net.blobs.items():
    # #     print(k,v.data.shape)
    # for param_tensor in network.state_dict():
    #     # print(str(i)+' |', param_tensor+' |', nameDict[param_tensor])
    #     if param_tensor not in pytorchLayerNameList:
    #         print("there is some problem in nameDict")
    #         sys.exit()

    #     param = network.state_dict()[param_tensor]
    #     print(param_tensor,param.cpu().data.numpy().shape)

    # [(k,v[0].data.shape) for k,v in net.params.items()]  #查看各层参数规模
    # w1=net.params['Convolution1'][0].data  #提取参数w
    # b1=net.params['Convolution1'][1].data  #提取参数b
    # net.forward()   #运行测试
    
    # [(k,v.data.shape) for k,v in net.blobs.items()]

    # caffe_Params = caffe_net.params

    # for k in sorted(caffe_Params):
    #     print(k)
    # print(len(caffe_Params))

    # for param_tensor in network.state_dict():
    #     print(str(i)+' |', param_tensor+' |', nameDict[param_tensor])
    #     if param_tensor not in pytorchLayerNameList:
    #         print("there is some problem in nameDict")
    #         sys.exit()

    #     param = network.state_dict()[param_tensor]
    #     # print('param.cpu().data.numpy():', param.cpu().data.numpy().shape)

    #     caffeLayerPara = nameDict[param_tensor]
    '''

    
    # # step 2: check the pytorch model weights and caffe model weights
    # caffenet = caffe.Net(protofile, modelFile, caffe.TEST)
    # # pytorchnet = build_ssd('test')
    # # pytorchnet.load_state_dict(torch.load(modelPath))
    # pytorchnet = torch.load(modelPath)
    # pytorchnet.eval()

    # with open('caffeVSpytoch_landmark.txt','w') as f:
    #     for param in nameDict.keys():
    #         print('param:', param)
    #         caffe_param = nameDict[param]
    #         print('caffe_param:', caffe_param)
    #         caffe_param_name,caffe_param_iter = caffe_param.split(',')

    #         parameters = caffenet.params[caffe_param_name][int(caffe_param_iter)].data[...]

    #         f.write(caffe_param_name + ' ' + caffe_param_iter)
    #         f.write('\n')
    #         f.write(str(parameters))
    #         f.write('\n')

    #         pytorch_param_name = param #nameDict[param]

    #         pytorch_parameters = pytorchnet.state_dict()[pytorch_param_name]

    #         f.write(str(pytorch_param_name))
    #         f.write('\n')
    #         f.write(str(pytorch_parameters))
    #         f.write('\n')
    