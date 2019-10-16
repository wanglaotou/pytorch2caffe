'''
@Author: Jiangtao
@Date: 2019-08-29 15:43:40
@LastEditors: Jiangtao
@LastEditTime: 2019-08-29 16:44:51
@Description: 
'''
import sys

import h5py

#sys.path.append("/home/workspace/yghan/caffe_lib/caffe/build/tools")
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

def max_pool(bottom, kernelSize=3, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, stride_h=stride[0], stride_w=stride[1], kernel_h=kernelSize[0], kernel_w=kernelSize[1])
    # return L.Pooling(bottom, pool=P.Pooling.MAX, global_pooling=True)

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
                                weight_filler={"type": "msra"},\
                                bias_term=False, param=dict(lr_mult=1))
    else:
        conv = caffe.layers.Convolution(bottom, num_output=outputChannel, pad=halfKernelSize, kernel_size=kernelSize, stride=stride, \
                                weight_filler={"type": "msra"},\
                                bias_term=False, param=dict(lr_mult=1)) 
    bn = BN(conv, isTrain=isTrain)
    if isRelu == True:
        return relu(bn)
    else:
        return bn


def generate_network(name,  intputSize=[17, 47, 1], writePath=None,isTrain=False):

    if name == 'pnet':
        net = caffe.NetSpec()
        net.data = L.Input(shape = dict(dim = [1,intputSize[2],intputSize[0],intputSize[1]]))

        # backbone
        net.conv1 = preluConv(net.data, 10, (3, 3), 1, isTrain, isRelu=True)     #64
        net.pool1 = max_pool(net.conv1, (3, 5), (3, 5))
        net.conv2 = preluConv(net.pool1, 16, (3, 5), 1, isTrain, isRelu=True)
        net.conv3 = preluConv(net.conv2, 32, (3, 5), 1, isTrain, isRelu=True)
        net.conv4_1 = preluConv(net.conv3, 2, (1, 1), 1, isTrain, isRelu=False)
        net.conv4_2 = preluConv(net.conv3, 4, (1, 1), 1, isTrain, isRelu=False)
    
    elif name == 'onet':
        net = caffe.NetSpec()
        net.data = L.Input(shape = dict(dim = [1,intputSize[2],intputSize[0],intputSize[1]]))

        # backbone
        net.conv1 = preluConv(net.data, 32, (3, 3), 1, isTrain, isRelu=True)     #64
        net.pool1 = max_pool(net.conv1, (3, 3), (2, 2))
        net.conv2 = preluConv(net.pool1, 64, (5, 3), 1, isTrain, isRelu=True)
        net.pool2 = max_pool(net.conv2, (3, 3), (2, 2))
        net.conv3 = preluConv(net.pool2, 64, (5, 3), 1, isTrain, isRelu=True)
        net.pool3 = max_pool(net.conv3, (2, 2), (2, 2))
        net.conv4 = preluConv(net.pool3, 128, (1, 1), 1, isTrain, isRelu=True)
        # print('net conv4', net.conv4)
        # print('net conv4', net.blobs['net.conv4'][0].data)
        # net.f1 = flatten()
        # net.f1= global_avg_pool(net.conv4)
        # # net.f1 = net.blobs['net.conv4'][0].data.reshape(32*3,5,5)
        net.fc1 = fully_connect(net.conv4, 256)
        net.d1 = drop(net.fc1, 0.25)
        net.prelu1 = prelu(net.d1)

        net.fc2 = fully_connect(net.prelu1, 2)
        net.fc3 = fully_connect(net.prelu1, 4)

    elif name == 'landmark':
        net = caffe.NetSpec()
        net.data = L.Input(shape = dict(dim = [1,intputSize[2],intputSize[0],intputSize[1]]))

        # backbone
        net.conv1 = basicConv(net.data, 32, 3, 2, isTrain)     #64
        net.conv2 = basicConv(net.conv1, 64, 3, 1, isTrain)
        net.conv3 = basicConv(net.conv2, 64, 3, 2, isTrain)
        net.conv4 = basicConv(net.conv3, 128, 3, 1, isTrain)     #64
        net.conv5 = basicConv(net.conv4, 128, 3, 2, isTrain)
        
        net.conv6 = basicConv(net.conv5, 128, 3, 1, isTrain)     #64
        net.conv7 = basicConv(net.conv6, 256, 3, 2, isTrain)
        net.conv8 = basicConv(net.conv7, 512, 3, 2, isTrain)
        net.gap1 = global_avg_pool(net.conv8)
        net.fc1 = fully_connect(net.gap1,8)

        
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
    # writePath = "onet.prototxt"
    if nettype == 'pnet':
        intputSize = [17, 47, 3]
    elif nettype == 'onet':
        intputSize = [34, 94, 3]
    elif nettype == 'landmark':
        intputSize = [128, 128, 1]
    # intputSize=[17, 47, 1]

    generate_network(nettype, intputSize=intputSize, writePath=writePath)
