'''
@Author: mrwang
@Date: 2019-08-29 15:43:40
@LastEditors: mrwang
@LastEditTime: 2019-08-29 16:44:51
@Description: 
'''
import sys

import h5py

import caffe
from caffe import layers as L
from caffe import params as P


def test(bottom):
    conv11 = caffe.layers.Convolution(bottom, num_output=20, kernel_size=3, weight_filler={"type": "xavier"},
                                bias_filler={"type": "constant"}, param=dict(lr_mult=1))
    relu11 = caffe.layers.PRelu(conv11, in_place=True)
    pool11 = caffe.layers.Pooling(relu11, pool=caffe.params.Pooling.MAX, kernel_size=3, stride=2)
    
    return pool11 

def relu(bottom):
    return L.ReLU(bottom, in_place=True)

def prelu(bottom):
    return L.PReLU(bottom, in_place=True)

def drop(bottom, dropout_ratio):
    return L.Dropout(bottom, dropout_ratio=0.25, in_place=True)

def fully_connect(bottom, outputChannel):
    return L.InnerProduct(bottom, num_output=outputChannel, weight_filler=dict(type='xavier'))

def flatten(net, bottom):
    return net.blobs['bottom'].data[0].flatten()


def global_avg_pool(bottom, kernelSize=3):
    #return L.Pooling(bottom, pool=P.Pooling.AVE,stride=1, kernel_size=kernelSize)
    return L.Pooling(bottom, pool=P.Pooling.AVE, global_pooling=True)

def preluConv(bottom, outputChannel, kernelSize=(3, 5), stride=1, isTrain=True, isRelu=True):
    if len(kernelSize) == 2:
        conv = caffe.layers.Convolution(bottom, num_output=outputChannel, kernel_h=kernelSize[0], kernel_w=kernelSize[1], stride=stride, \
                                    weight_filler={"type": "xavier"},\
                                    bias_term=True, param=dict(lr_mult=1))
    elif len(kernelSize) == 1:
        conv = caffe.layers.Convolution(bottom, num_output=outputChannel, kernel_size=kernelSize[0], stride=stride, \
                                    weight_filler={"type": "xavier"},\
                                    bias_term=True, param=dict(lr_mult=1))
    else:
        pass
    if isRelu == True:
        return prelu(conv)
    else:
        return conv

def strpreluConv(bottom, outputChannel, kernelSize=3, stride=(1, 2), isTrain=True, isRelu=True):
    if len(stride) == 1:
        conv = caffe.layers.Convolution(bottom, num_output=outputChannel, kernel_size=kernelSize, stride=stride, \
                                    weight_filler={"type": "xavier"},\
                                    bias_term=True, param=dict(lr_mult=1))
    elif len(stride) == 2:
        conv = caffe.layers.Convolution(bottom, num_output=outputChannel, kernel_size=kernelSize, stride_h=stride[0], stride_w=stride[1], \
                                    weight_filler={"type": "xavier"},\
                                    bias_term=True, param=dict(lr_mult=1))
    else:
        pass
    if isRelu == True:
        return prelu(conv)
    else:
        return conv

def max_pool(bottom, kernelSize=3, stride=2):
    # return L.Pooling(bottom, pool=P.Pooling.MAX, stride_h=stride[0], stride_w=stride[1], kernel_h=kernelSize[0], kernel_w=kernelSize[1])
    return L.Pooling(bottom, pool=P.Pooling.MAX, stride=stride, kernel_size=kernelSize)
    # return L.Pooling(bottom, pool=P.Pooling.MAX, global_pooling=True)

# def BN(bottom, isTrain=True, isRelu=False):
#     use_global = False
#     if isTrain == False:
#         use_global=True
#     bn = caffe.layers.BatchNorm(bottom, use_global_stats=use_global, in_place=True)
#     scale = caffe.layers.Scale(bn, bias_term=True, in_place=True)
#     return scale
#     # if True == isRelu:
#     #     return relu(scale)
#     # else:
#     #     return scale

def BN(bottom, isTrain=True, isRelu=False):
    use_global = False
    if isTrain == False:
        use_global=True
    bn = caffe.layers.BatchNorm(bottom, use_global_stats=use_global, in_place=True)
    scale = caffe.layers.Scale(bn, bias_term=True, in_place=True)
    if True == isRelu:
        return relu(scale)
    else:
        return scale

def basicConv(bottom, outputChannel, kernelSize=3, stride=2, isTrain=True, isRelu=True):
    halfKernelSize = int(kernelSize/2)
    #print("half", halfKernelSize)
    if kernelSize == 1:
        halfKernelSize=0
    if halfKernelSize == 0:
        conv = caffe.layers.Convolution(bottom, num_output=outputChannel, kernel_size=kernelSize, stride=stride, \
                                weight_filler={"type": "xavier"},\
                                bias_term=False, param=dict(lr_mult=1))
    else:
        conv = caffe.layers.Convolution(bottom, num_output=outputChannel, pad=halfKernelSize, kernel_size=kernelSize, stride=stride, \
                                weight_filler={"type": "xavier"},\
                                bias_term=False, param=dict(lr_mult=1)) 
    bn = BN(conv, isTrain=isTrain)
    if isRelu == True:
        return relu(bn)
    else:
        return bn


## ==================== landmark  ==========================
def lmark_conv(bottom, outputChannel, kernel_size=3, stride=2, isTrain=False, isRelu=True):
    pad = int(kernel_size/2)
    conv = caffe.layers.Convolution(bottom, num_output=outputChannel, kernel_size=kernel_size, stride=stride, \
                                pad=pad, weight_filler={"type": "msra"},\
                                bias_term=False, param=dict(lr_mult=1))
 
    bn = BN(conv, isTrain=isTrain)
    if isRelu == True:
        return relu(bn)
    else:
        return bn

def lmark_global_avgpool(bottom, kernelSize=3):
    #return L.Pooling(bottom, pool=P.Pooling.AVE,stride=1, kernel_size=kernelSize)
    return L.Pooling(bottom, pool=P.Pooling.AVE, global_pooling=True)

def lmark_fully_connect(bottom, outputChannel):
    return L.InnerProduct(bottom, num_output=outputChannel, weight_filler=dict(type='xavier'))

## ====================  ssd mobilenet  ==========================
def conv_bn(name, input, output, kernel_size=3, stride=1, pad=1, activation=True, dilation=1):
    conv = L.Convolution(input, kernel_size=kernel_size, stride=stride, num_output=output, bias_term=False, pad=pad, weight_filler=dict(type='xavier'), dilation=dilation)

    # in-place compute means your input and output has the same memory area,which will be more memory effienct
    bn = L.BatchNorm(conv, use_global_stats=True, in_place=True)

    # scale = L.Scale(bn,filler=dict(value=1),bias_filler=dict(value=0),bias_term=True, in_place=True)
    out = L.Scale(bn, bias_term=True, in_place=True)

    if activation is True:
        out = L.ReLU(out, in_place=True)
    return out

def conv(input, output, kernel_size=3, stride=1, pad=1, activation=True):
    out = L.Convolution(input, kernel_size=kernel_size, stride=stride, num_output=output, bias_term=True, pad=pad, weight_filler=dict(type='xavier'))  # ,name = name

    if activation is True:
        out = L.ReLU(out, in_place=True)
    return out

def conv_fpn(input, output, kernel_size=3, stride=1, pad=1, activation=False):
    out = L.Convolution(input, kernel_size=kernel_size, stride=stride, num_output=output, bias_term=True, pad=pad, weight_filler=dict(type='xavier'))  # ,name = name

    if activation is True:
        out = L.ReLU(out, in_place=True)
    return out

def deconv(input, output, kernel_size=3, stride=2, pad=1, activation=True):
    out = L.Deconvolution(input, convolution_param=dict(kernel_size=kernel_size, stride=stride, num_output=output, pad=pad), param=[dict(lr_mult=0)])

    if activation is True:
        out = L.ReLU(out, in_place=True)
    return out

def eltwise(input, branch_shortcut):
    out = L.Eltwise(input, branch_shortcut, eltwise_param=dict(operation=1))
    return out

def mobconv1_bn_relu(input, output, kernel_size=1, stride=1, pad=0, activation=True):
    conv1 = L.Convolution(input, kernel_size=1, stride=1, num_output=output, bias_term=False, pad=pad, weight_filler=dict(type='xavier'))
    bn1 = L.BatchNorm(conv1, use_global_stats=True, in_place=True)
    scale1 = L.Scale(bn1, bias_term=True, in_place=True)
    out = L.ReLU(scale1, in_place=True)
    return out

def mobconv3_bn(input, output, kernel_size=3, stride=1, pad=1, group=1, activation=True):
    conv1 = L.Convolution(input, kernel_size=3, stride=stride, num_output=output, bias_term=False, pad=pad, group=group, weight_filler=dict(type='xavier'))
    bn1 = L.BatchNorm(conv1, use_global_stats=True, in_place=True)
    scale1 = L.Scale(bn1, bias_term=True, in_place=True)
    out = L.ReLU(scale1, in_place=True)
    return out

def mobconv1_bn(input, output, kernel_size=1, stride=1, pad=0, activation=False):
    conv1 = L.Convolution(input, kernel_size=1, stride=1, num_output=output, bias_term=False, pad=pad, weight_filler=dict(type='xavier'))
    bn1 = L.BatchNorm(conv1, use_global_stats=True, in_place=True)
    out = L.Scale(bn1, bias_term=True, in_place=True)
    return out

def mobconv1(input, output, kernel_size=1, stride=1, pad=0, activation=False):
    conv1 = L.Convolution(input, kernel_size=1, stride=1, num_output=output, bias_term=False, pad=pad, weight_filler=dict(type='xavier'))
    out = conv1
    return out


# def linear_bottleneck(input, output, kernel_size=3, stride=1, pad=1, t=1, group=1, activation=True):
#     residual = input
#     conv1 = mobconv1_bn_relu(input, input*t, kernel_size=1, stride=1, pad=0, activation=True)
#     conv2 = mobconv3_bn(conv1, input*t, input*t, kernel_size=3, stride=stride, pad=1, group=1, activation=True)
#     out = mobconv1_bn(conv2, input*t, output, kernel_size=1, stride=1, pad=0, activation=False)
#     if(stride==1 and input==output):
#         out += residual
#     return out

def mobilenet_conv_first(input, output, kernel_size=3, stride=2, pad=1, activation=True):
    conv1 = L.Convolution(input, kernel_size=3, stride=2, num_output=output, bias_term=False, pad=pad, weight_filler=dict(type='xavier'))
    bn1 = L.BatchNorm(conv1, use_global_stats=True, in_place=True)
    scale1 = L.Scale(bn1, bias_term=True, in_place=True)
    out = L.ReLU(scale1, in_place=True)
    return out

def mobilenet_conv_last(input, output, kernel_size=1, stride=1, pad=0, activation=True):
    conv1 = L.Convolution(input, kernel_size=1, stride=1, num_output=output, bias_term=False, pad=pad, weight_filler=dict(type='xavier'))
    bn1 = L.BatchNorm(conv1, use_global_stats=True, in_place=True)
    scale1 = L.Scale(bn1, bias_term=True, in_place=True)
    out = L.ReLU(scale1, in_place=True)
    return out

def extra_conv1(input, output, kernel_size=1, stride=1, pad=0, activation=True):
    conv1 = L.Convolution(input, kernel_size=1, stride=1, num_output=output, bias_term=False, pad=pad, weight_filler=dict(type='xavier'))
    bn1 = L.BatchNorm(conv1, use_global_stats=True, in_place=True)
    scale1 = L.Scale(bn1, bias_term=True, in_place=True)
    out = L.ReLU(scale1, in_place=True)
    return out

def extra_conv3(input, output, kernel_size=3, stride=2, pad=1, activation=True):
    conv1 = L.Convolution(input, kernel_size=3, stride=2, num_output=output, bias_term=False, pad=pad, weight_filler=dict(type='xavier'))
    bn1 = L.BatchNorm(conv1, use_global_stats=True, in_place=True)
    scale1 = L.Scale(bn1, bias_term=True, in_place=True)
    out = L.ReLU(scale1, in_place=True)
    return out

def loc_conf_conv(input, output, kernel_size=1, stride=1, pad=0, activation=False):
    out = L.Convolution(input, kernel_size=1, stride=1, num_output=output, bias_term=True, pad=pad, weight_filler=dict(type='xavier'))
    return out


def generate_network(name,  intputSize=[300, 300, 3], writePath=None,isTrain=False):

    net = caffe.NetSpec()
    net.data = L.Input(shape = dict(dim = [1,intputSize[2],intputSize[0],intputSize[1]]))

    if name == 'ssd_mobilenetv2_fpn':
                
        # # backbone
        base_channel = int(16)
        net.conv0 = mobilenet_conv_first(net.data, base_channel * 2, kernel_size=3, stride=2, pad=1)  # output: 32 * 150 * 150
        
##Bottleneck_0
        net.conv1 = mobconv1_bn_relu(net.conv0, base_channel * 2, kernel_size=1, stride=1, pad=0)     # output: 32   *150*150
        net.conv2 = mobconv3_bn(net.conv1, base_channel * 2, kernel_size=3, stride=1, pad=1, group=base_channel * 2)  # 32  *150*150  stride=1
        net.conv3 = mobconv1_bn(net.conv2, base_channel, kernel_size=1, stride=1, pad=0)   # 16
        # net.elt1 = eltwise(net.conv0, net.conv3)       ## input != output  
        
##Bottleneck_1
        net.conv4 = mobconv1_bn_relu(net.conv3, base_channel * 6, kernel_size=1, stride=1, pad=0)  # output: 96
        net.conv5 = mobconv3_bn(net.conv4, base_channel * 6, kernel_size=3, stride=2, pad=1, group=base_channel * 6)  # 96  *75*75
        net.conv6 = mobconv1_bn(net.conv5, int(base_channel * 3/2), kernel_size=1, stride=1, pad=0)  # 24
        net.conv7 = mobconv1_bn_relu(net.conv6, base_channel * 9, kernel_size=1, stride=1, pad=0)  # 144
        net.conv8 = mobconv3_bn(net.conv7, base_channel * 9, kernel_size=3, stride=1, pad=1, group=base_channel * 9)  # 144
        net.conv9 = mobconv1_bn(net.conv8, int(base_channel * 3/2), kernel_size=1, stride=1, pad=0)  # 24
        net.elt1 = eltwise(net.conv6, net.conv9)        # --->  ok!!!  24 *75*75  
        
##Bottleneck_2
        net.conv10 = mobconv1_bn_relu(net.elt1, base_channel * 9, kernel_size=1, stride=1, pad=0)  # 144
        net.conv11 = mobconv3_bn(net.conv10, base_channel * 9, kernel_size=3, stride=2, pad=1, group=base_channel * 9)  # 144   *38*38
        net.conv12 = mobconv1_bn(net.conv11, base_channel * 2, kernel_size=1, stride=1, pad=0)  # 32
        net.conv13 = mobconv1_bn_relu(net.conv12, base_channel * 12, kernel_size=1, stride=1, pad=0)  # 192
        net.conv14 = mobconv3_bn(net.conv13, base_channel * 12, kernel_size=3, stride=1, pad=1, group=base_channel * 12)  # 192
        net.conv15 = mobconv1_bn(net.conv14, base_channel * 2, kernel_size=1, stride=1, pad=0)  # 32
        net.elt2 = eltwise(net.conv12, net.conv15)
        net.conv16 = mobconv1_bn_relu(net.elt2, base_channel * 12, kernel_size=1, stride=1, pad=0)  # 192    
        net.conv17 = mobconv3_bn(net.conv16, base_channel * 12, kernel_size=3, stride=1, pad=1, group=base_channel * 12)  # 192   ok
             
        net.conv18 = mobconv1_bn(net.conv17, base_channel * 2, kernel_size=1, stride=1, pad=0)  # 32  不一样
        # net.conv18 = mobconv1(net.conv17, base_channel * 2, kernel_size=1, stride=1, pad=0)
        
        net.elt3 = eltwise(net.elt2, net.conv18)      ##  -->  not ok!!!  32 * 38*38
        
##Bottleneck_3
        net.conv19 = mobconv1_bn_relu(net.elt3, base_channel * 12, kernel_size=1, stride=1, pad=0)  # 192
        net.conv20 = mobconv3_bn(net.conv19, base_channel * 12, kernel_size=3, stride=2, pad=1, group=base_channel * 12)  # 192 *19*19
        net.conv21 = mobconv1_bn(net.conv20, base_channel * 4, kernel_size=1, stride=1, pad=0)  # 64
        net.conv22 = mobconv1_bn_relu(net.conv21, base_channel * 24, kernel_size=1, stride=1, pad=0)  # 384
        net.conv23 = mobconv3_bn(net.conv22, base_channel * 24, kernel_size=3, stride=1, pad=1, group=base_channel * 24)  # 384
        net.conv24 = mobconv1_bn(net.conv23, base_channel * 4, kernel_size=1, stride=1, pad=0)  # 64
        net.elt4 = eltwise(net.conv21, net.conv24)
        net.conv25 = mobconv1_bn_relu(net.elt4, base_channel * 24, kernel_size=1, stride=1, pad=0)  # 384
        net.conv26 = mobconv3_bn(net.conv25, base_channel * 24, kernel_size=3, stride=1, pad=1, group=base_channel * 24)  # 384
        net.conv27 = mobconv1_bn(net.conv26, base_channel * 4, kernel_size=1, stride=1, pad=0)  # 64
        net.elt5 = eltwise(net.elt4, net.conv27)
        net.conv28 = mobconv1_bn_relu(net.elt5, base_channel * 24, kernel_size=1, stride=1, pad=0)  # 384
        net.conv29 = mobconv3_bn(net.conv28, base_channel * 24, kernel_size=3, stride=1, pad=1, group=base_channel * 24)  # 384
        net.conv30 = mobconv1_bn(net.conv29, base_channel * 4, kernel_size=1, stride=1, pad=0)  # 64
        net.elt6 = eltwise(net.elt5, net.conv30)              # -->  not ok!!!  64 * 19*19
        
##Bottleneck_4
        net.conv31 = mobconv1_bn_relu(net.elt6, base_channel * 24, kernel_size=1, stride=1, pad=0)  # 384
        net.conv32 = mobconv3_bn(net.conv31, base_channel * 24, kernel_size=3, stride=1, pad=1, group=base_channel * 24)  # 384   *19*19  stride=1
        # net.conv33 = mobconv1_bn(net.conv32, base_channel * 6, kernel_size=1, stride=1, pad=0)  # 96
        net.conv33 = mobconv1_bn(net.conv32, base_channel * 8, kernel_size=1, stride=1, pad=0)  # 128
        ## error this block
        net.conv34 = mobconv1_bn_relu(net.conv33, base_channel * 48, kernel_size=1, stride=1, pad=0)  # 576     ## 36 -> 48, change this channel ok....
        
        net.conv35 = mobconv3_bn(net.conv34, base_channel * 48, kernel_size=3, stride=1, pad=1, group=base_channel * 48)  # 576
        # net.conv36 = mobconv1_bn(net.conv35, base_channel * 6, kernel_size=1, stride=1, pad=0)  # 96
        net.conv36 = mobconv1_bn(net.conv35, base_channel * 8, kernel_size=1, stride=1, pad=0)  # 128
        net.elt7 = eltwise(net.conv33, net.conv36)
        
        net.conv37 = mobconv1_bn_relu(net.elt7, base_channel * 48, kernel_size=1, stride=1, pad=0)  # 576       ## 36 -> 48, change this channel ok....
        net.conv38 = mobconv3_bn(net.conv37, base_channel * 48, kernel_size=3, stride=1, pad=1, group=base_channel * 48)  # 576
        # net.conv39 = mobconv1_bn(net.conv38, base_channel * 6, kernel_size=1, stride=1, pad=0)  # 96    *19*19
        net.conv39 = mobconv1_bn(net.conv38, base_channel * 8, kernel_size=1, stride=1, pad=0)  # 128    *19*19      
        net.elt8 = eltwise(net.elt7, net.conv39)        # 128    *19*19
        # net.elt8 = eltwise(net.elt7, net.deconv1)
        
##Bottleneck_5
        # net.conv40 = mobconv1_bn_relu(net.elt8, base_channel * 36, kernel_size=1, stride=1, pad=0)  # 576 -> 768    ## 36 -> 48, change this channel ok...
        net.conv40 = mobconv1_bn_relu(net.elt8, base_channel * 48, kernel_size=1, stride=1, pad=0)  # 576
        net.conv41 = mobconv3_bn(net.conv40, base_channel * 48, kernel_size=3, stride=2, pad=1, group=base_channel * 48)  # 576   *10*10
        net.conv42 = mobconv1_bn(net.conv41, base_channel * 10, kernel_size=1, stride=1, pad=0)  # 160
        net.conv43 = mobconv1_bn_relu(net.conv42, base_channel * 60, kernel_size=1, stride=1, pad=0)  # 960
        net.conv44 = mobconv3_bn(net.conv43, base_channel * 60, kernel_size=3, stride=1, pad=1, group=base_channel * 60)  # 960
        net.conv45 = mobconv1_bn(net.conv44, base_channel * 10, kernel_size=1, stride=1, pad=0)  # 160
        net.elt9 = eltwise(net.conv42, net.conv45)
        net.conv46 = mobconv1_bn_relu(net.elt9, base_channel * 60, kernel_size=1, stride=1, pad=0)  # 960
        net.conv47 = mobconv3_bn(net.conv46, base_channel * 60, kernel_size=3, stride=1, pad=1, group=base_channel * 60)  # 960
        net.conv48 = mobconv1_bn(net.conv47, base_channel * 10, kernel_size=1, stride=1, pad=0)  # 160
        net.elt10 = eltwise(net.elt9, net.conv48)
        
##Bottleneck_6
        net.conv49 = mobconv1_bn_relu(net.elt10, base_channel * 60, kernel_size=1, stride=1, pad=0)     # 960
        net.conv50 = mobconv3_bn(net.conv49, base_channel * 60, kernel_size=3, stride=1, pad=1, group=base_channel * 60)  # 960   *10*10  stride=1
        net.conv51 = mobconv1_bn(net.conv50, base_channel * 20, kernel_size=1, stride=1, pad=0)  # 320  ×10*10
             
## fpn  
        net.fpn1 = conv_fpn(net.conv51, base_channel * 8, kernel_size=1, stride=1, pad=0)      ## add for fpn, C5
        net.deconv1 = deconv(net.fpn1, base_channel * 8, kernel_size=3, stride=2, pad=1)
        net.fpn2 = conv_fpn(net.fpn1, base_channel * 8, kernel_size=3, stride=1, pad=1)

        net.fpn3 = conv_fpn(net.elt8, base_channel * 8, kernel_size=1, stride=1, pad=0)
        net.fpnelt = eltwise(net.deconv1, net.fpn3)
        net.fpn4 = conv_fpn(net.fpnelt, base_channel * 8, kernel_size=3, stride=1, pad=1)

        # net.elt7 = eltwise(net.conv53, net.conv55)
        net.conv52 = mobilenet_conv_last(net.conv51, base_channel * 8, kernel_size=1, stride=1, pad=0, activation=True)
        # net.conv52 = mobilenet_conv_last(net.fpn2, base_channel * 80, kernel_size=1, stride=1, pad=0, activation=True)
        ## --> ok
##Extras_layer
        net.conv_ex1 = extra_conv1(net.conv52, base_channel * 16, kernel_size=1, stride=1, pad=0)    #256
        net.conv_ex2 = extra_conv3(net.conv_ex1, base_channel * 32, kernel_size=3, stride=2, pad=1)  #512   *5*5

        net.conv_ex3 = extra_conv1(net.conv_ex2, base_channel * 8, kernel_size=1, stride=1, pad=0)  #128
        net.conv_ex4 = extra_conv3(net.conv_ex3, base_channel * 16, kernel_size=3, stride=2, pad=1)  #256   *3*3

        net.conv_ex5 = extra_conv1(net.conv_ex4, base_channel * 8, kernel_size=1, stride=1, pad=0)  #128
        net.conv_ex6 = extra_conv3(net.conv_ex5, base_channel * 16, kernel_size=3, stride=2, pad=1)  #256   *2*2

        net.conv_ex7 = extra_conv1(net.conv_ex6, base_channel * 4, kernel_size=1, stride=1, pad=0)  #64
        net.conv_ex8 = extra_conv3(net.conv_ex7, base_channel * 8, kernel_size=3, stride=2, pad=1)  #128    *1*1
        
##loc and conf layer
        # net.loc1 = loc_conf_conv(net.elt8, base_channel, kernel_size=1, stride=1, pad=0)                 #96 -> 16
        # net.conf1 = loc_conf_conv(net.elt8, int(base_channel * 1/2), kernel_size=1, stride=1, pad=0)    #96 -> 8
        net.loc1 = loc_conf_conv(net.fpn4, base_channel, kernel_size=1, stride=1, pad=0)                 #96 -> 16
        net.conf1 = loc_conf_conv(net.fpn4, int(base_channel * 1/2), kernel_size=1, stride=1, pad=0)    #96 -> 8

        # net.loc2 = loc_conf_conv(net.conv52, int(base_channel * 3/2), kernel_size=1, stride=1, pad=0)    #1280 -> 24
        # net.conf2 = loc_conf_conv(net.conv52, int(base_channel * 3/4), kernel_size=1, stride=1, pad=0)  #1280 -> 12
        net.loc2 = loc_conf_conv(net.fpn2, int(base_channel * 3/2), kernel_size=1, stride=1, pad=0)    #1280 -> 24
        net.conf2 = loc_conf_conv(net.fpn2, int(base_channel * 3/4), kernel_size=1, stride=1, pad=0)  #1280 -> 12

        net.loc3 = loc_conf_conv(net.conv_ex2, int(base_channel * 3/2), kernel_size=1, stride=1, pad=0)    #512 -> 24
        net.conf3 = loc_conf_conv(net.conv_ex2, int(base_channel * 3/4), kernel_size=1, stride=1, pad=0)   #512 -> 12

        net.loc4 = loc_conf_conv(net.conv_ex4, int(base_channel * 3/2), kernel_size=1, stride=1, pad=0)    #256 -> 24
        net.conf4 = loc_conf_conv(net.conv_ex4, int(base_channel * 3/4), kernel_size=1, stride=1, pad=0)   #256 -> 12

        net.loc5 = loc_conf_conv(net.conv_ex6, int(base_channel * 3/2), kernel_size=1, stride=1, pad=0)    #256 -> 24
        net.conf5 = loc_conf_conv(net.conv_ex6, int(base_channel * 3/4), kernel_size=1, stride=1, pad=0)   #256 -> 12

        net.loc6 = loc_conf_conv(net.conv_ex8, int(base_channel * 3/2), kernel_size=1, stride=1, pad=0)    #128 -> 24
        net.conf6 = loc_conf_conv(net.conv_ex8, int(base_channel * 3/4), kernel_size=1, stride=1, pad=0)   #128 -> 12
    
    if name == 'landmark':
        base_channel = int(32)

        net.conv1 = lmark_conv(net.data, base_channel * 1, kernel_size=3, stride=2)
        net.conv2 = lmark_conv(net.conv1, base_channel * 2, kernel_size=3, stride=1)
        net.conv3 = lmark_conv(net.conv2, base_channel * 2, kernel_size=3, stride=2)
        net.conv4 = lmark_conv(net.conv3, base_channel * 4, kernel_size=3, stride=1)
        net.conv5 = lmark_conv(net.conv4, base_channel * 4, kernel_size=3, stride=2)

        net.conv6 = lmark_conv(net.conv5, base_channel * 4, kernel_size=3, stride=1)
        net.conv7 = lmark_conv(net.conv6, base_channel * 8, kernel_size=3, stride=2)
        net.conv8 = lmark_conv(net.conv7, base_channel * 16, kernel_size=3, stride=2)
        
        net.gap1 = lmark_global_avgpool(net.conv8)
        net.fc22 = lmark_fully_connect(net.gap1, 8)
        
    # transform
    proto = net.to_proto()
    proto.name = name

    with open(writePath, 'w') as f:
        print("start write!\n")
        f.write(str(proto))

    # 检查参数名
    net = caffe.Net(writePath, caffe.TEST)
    caffeParams = net.params
    for k in sorted(caffeParams):
        print(k)
    print(len(caffeParams))

if __name__ == '__main__':

    nettype = 'landmark' # 'pnet, onet, landmark'
    writePath = nettype + '.prototxt'
    if nettype == 'pnet':
        intputSize = [17, 47, 3]
    elif nettype == 'ssd_mobilenetv2_fpn':
        intputSize = [300, 300, 3]
    elif nettype == 'landmark':
        intputSize = [128, 128, 3]

    generate_network(nettype, intputSize=intputSize, writePath=writePath)
