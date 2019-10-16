'''
@Author: Jiangtao
@Date: 2019-08-07 10:42:06
@LastEditors: Jiangtao
@LastEditTime: 2019-08-20 14:06:38
@Description: 
'''
#
#   Darknet related layers
#   Copyright EAVISE
#

# modified by mileiston

import logging as log
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

#import util 

__all__ = ['Conv2dBatchLeaky', 'Conv2dBatch', 'GlobalAvgPool2d', 'PaddedMaxPool2d', 'Reorg', 'SELayer',
            'CReLU', 'Scale', 'ScaleReLU', 'L2Norm', 'Conv2dL2NormLeaky', 'PPReLU', 'Conv2dBatchPPReLU',
            'Conv2dBatchPReLU', 'Conv2dBatchPLU', 'Conv2dBatchELU', 'Conv2dBatchSELU',
            'Shuffle', 'Conv2dBatchReLU', 'FullyConnectLayer', 'Conv2d1x1', 'Conv2dDepthWise', 'buildInvertedResBlock',
            'layerConcat', 'Conv2d1x1Relu6','ConvTranspose2dBatchReLU']

class Conv2dBatchLeaky(nn.Module):
    """ This convenience layer groups a 2D convolution, a batchnorm and a leaky ReLU.
    They are executed in a sequential manner.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the kernel of the convolution
        stride (int or tuple): Stride of the convolution
        padding (int or tuple): padding of the convolution
        leaky_slope (number, optional): Controls the angle of the negative slope of the leaky ReLU; Default **0.1**
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, leaky_slope=0.1):
        super(Conv2dBatchLeaky, self).__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if isinstance(kernel_size, (list, tuple)):
            self.padding = [int(ii/2) for ii in kernel_size]
        else:
            self.padding = int(kernel_size/2)
        self.leaky_slope = leaky_slope

        # Layer
        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False),
            nn.BatchNorm2d(self.out_channels), #, eps=1e-6, momentum=0.01),
            nn.LeakyReLU(self.leaky_slope, inplace=True)
        )

    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, negative_slope={leaky_slope})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
        x = self.layers(x)
        return x


class Conv2dBatchPPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if isinstance(kernel_size, (list, tuple)):
            self.padding = [int(ii/2) for ii in kernel_size]
        else:
            self.padding = int(kernel_size/2)

        # Layer
        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False),
            nn.BatchNorm2d(self.out_channels), #, eps=1e-6, momentum=0.01),
            PPReLU(self.out_channels)
        )

    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
        x = self.layers(x)
        return x

class Conv2d1x1Relu6(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, isBias = False):
        super().__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = 1
        self.stride = stride
        self.isBias = isBias
        # Layer
        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, bias=self.isBias),
            nn.BatchNorm2d(self.out_channels), #
            nn.ReLU(inplace=True)
        )

    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
       # print(x.size())
        #print("xinput", x)
        x = self.layers(x)
       # print("after: ", x.size())
        #print("************************* Conv2d1x1Relu6 weight beign")
       # print(self.layers[2])

        #print(self.layers[0].weight)
        #print(self.layers[1].weight)
        #print(self.layers[1].bias)
        #print(self.layers[1].running_mean)
       # print(self.layers[1].running_var)
       # print(self.layers[2])

       # print("xout", x)
        return x

    def toCaffe():
        weightList = util.annalysis_pytorch_layer_name(self.layers, isBias=False)
        layerNameList = ["one"]

        return layerNameList, weightList
class Conv2d1x1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = 1
        self.stride = stride

        # Layer
        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, bias=False),
        )

    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
        #print(x.size())
        x = self.layers(x)
       # print("after: ", x.size())
        return x

class Conv2dBatchPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if isinstance(kernel_size, (list, tuple)):
            self.padding = [int(ii/2) for ii in kernel_size]
        else:
            self.padding = int(kernel_size/2)

        # Layer
        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False),
            nn.BatchNorm2d(self.out_channels), #, eps=1e-6, momentum=0.01),
            nn.PReLU(self.out_channels)
        )

    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
        x = self.layers(x)
        return x


class Conv2dBatchPLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if isinstance(kernel_size, (list, tuple)):
            self.padding = [int(ii/2) for ii in kernel_size]
        else:
            self.padding = int(kernel_size/2)

        # Layer
        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False),
            nn.BatchNorm2d(self.out_channels), #, eps=1e-6, momentum=0.01),
            PLU()
        )

    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
        y = self.layers(x)
        return y


class Conv2dBatchELU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = int(kernel_size/2)
        if isinstance(kernel_size, (list, tuple)):
            self.padding = [int(ii/2) for ii in kernel_size]
        else:
            self.padding = int(kernel_size/2)

        # Layer
        self.layer = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False),
            nn.BatchNorm2d(self.out_channels), #, eps=1e-6, momentum=0.01),
            nn.ELU(inplace=True)
        )

    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
        y = self.layer(x)
        return y


class Conv2dBatchSELU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if isinstance(kernel_size, (list, tuple)):
            self.padding = [int(ii/2) for ii in kernel_size]
        else:
            self.padding = int(kernel_size/2)

        # Layer
        self.layer = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False),
            nn.BatchNorm2d(self.out_channels), #, eps=1e-6, momentum=0.01),
            nn.SELU(inplace=True)
        )

    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
        y = self.layer(x)
        return y


class Conv2dBatch(nn.Module):
    """ This convenience layer groups a 2D convolution, a batchnorm and a leaky ReLU.
    They are executed in a sequential manner.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the kernel of the convolution
        stride (int or tuple): Stride of the convolution
        padding (int or tuple): padding of the convolution
        leaky_slope (number, optional): Controls the angle of the negative slope of the leaky ReLU; Default **0.1**
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, leaky_slope=0.1):
        super().__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if isinstance(kernel_size, (list, tuple)):
            self.padding = [int(ii/2) for ii in kernel_size]
        else:
            self.padding = int(kernel_size/2)
        self.leaky_slope = leaky_slope

        # Layer
        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )

    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
        x = self.layers(x)
        return x


class GlobalAvgPool2d(nn.Module):
    """ This layer averages each channel to a single number.
    """
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        #print("gapSize: ", B, C, H, W, x.size())
        x = x.view(B, C)
        #print("gap2d", x.size())
        return x

    def toCaffe():
        weightList = util.annalysis_pytorch_layer_name("gap")
        layerNameList = ["one"]

        return layerNameList, weightList    

class FullyConnectLayer(nn.Module):
    """ This layer averages each channel to a single number.
    """
    def __init__(self, in_channels, out_channels):
        super(FullyConnectLayer, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.layers = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
        )

    def forward(self, x):
        if len(x.size()) < 2:
            print("FullyConnectLayer input error!\n")
            sys.exit()
        flattenNum = 1
        for i in range(1,len(x.size())): 
            flattenNum *= x.size(i)  

        x = x.view(-1, flattenNum)
        x = self.layers(x)  
        return x
        #print(x, x.data)
        #print(x)
        #print(x.data)
        #print("fcn: ", B, C, H, W, x.size())
        #return torch.FloatTensor(x)

    def toCaffe():
        weightList = util.annalysis_pytorch_layer_name(self.layers, True)
        layerNameList = ["one"]
        return layerNameList, weightList    

class PaddedMaxPool2d(nn.Module):
    """ Maxpool layer with a replicating padding.

    Args:
        kernel_size (int or tuple): Kernel size for maxpooling
        stride (int or tuple, optional): The stride of the window; Default ``kernel_size``
        padding (tuple, optional): (left, right, top, bottom) padding; Default **None**
        dilation (int or tuple, optional): A parameter that controls the stride of elements in the window
    """
    def __init__(self, kernel_size, stride=None, padding=(0, 0, 0, 0), dilation=1):
        super(PaddedMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation

    def __repr__(self):
        return f('{self.__class__.__name__} (kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation})')

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, self.padding, mode='replicate'), self.kernel_size, self.stride, 0, self.dilation)
        return x


class Reorg(nn.Module):
    """ This layer reorganizes a tensor according to a stride.
    The dimensions 2,3 will be sliced by the stride and then stacked in dimension 1. (input must have 4 dimensions)

    Args:
        stride (int): stride to divide the input tensor
    """
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        if not isinstance(stride, int):
            raise TypeError(f('stride is not an int [{type(stride)}]'))
        self.stride = stride
        self.darknet = True

    def __repr__(self):
        return f('{self.__class__.__name__} (stride={self.stride}, darknet_compatible_mode={self.darknet})')

    def forward(self, x):
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)

        if H % self.stride != 0:
            raise ValueError(f('Dimension mismatch: {H} is not divisible by {self.stride}'))
        if W % self.stride != 0:
            raise ValueError(f('Dimension mismatch: {W} is not divisible by {self.stride}'))

        # darknet compatible version from: https://github.com/thtrieu/darkflow/issues/173#issuecomment-296048648
     
        if self.darknet:
            #print("xBefore", x.shape)
            x = x.view(B, C//(self.stride**2), H, self.stride, W, self.stride).contiguous()
            #print(x.shape)
            x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
            x = x.view(B, -1, H//self.stride, W//self.stride)
           # print(x.shape)
        else:
            ws, hs = self.stride, self.stride
            x = x.view(B, C, H//hs, hs, W//ws, ws).transpose(3, 4).contiguous()
            x = x.view(B, C, H//hs*W//ws, hs*ws).transpose(2, 3).contiguous()
            x = x.view(B, C, hs*ws, H//hs, W//ws).transpose(1, 2).contiguous()
            x = x.view(B, hs*ws*C, H//hs, W//ws)

        return x


class SELayer(nn.Module):
    def __init__(self, nchannels, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(nchannels, nchannels // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(nchannels // reduction, nchannels),
                nn.Sigmoid()
        )
        self.nchannels = nchannels
        self.reudction = reduction

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

    def __repr__(self):
        s = '{name} ({nchannels}, {reduction})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class Scale(nn.Module):
    def __init__(self, nchannels, bias=True, init_scale=1.0):
        super().__init__()
        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.nchannels = nchannels
        self.weight = nn.Parameter(torch.Tensor(1, nchannels, 1, 1))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, nchannels, 1, 1))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        # Not a very smart way to initialize weights
        self.reset_parameters(init_scale)

    def reset_parameters(self, init_scale=1.0):
        self.weight.data.fill_(init_scale)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, x):
        # See the autograd section for explanation of what happens here.
        y = x * self.weight
        if self.bias is not None:
            y += self.bias
        return y

    def __repr__(self):
        s = '{} ({}, {})'
        return s.format(self.__class__.__name__, self.nchannels, self.bias is not None)


class ScaleReLU(nn.Module):
    def __init__(self, nchannels):
        super().__init__()
        self.scale = Scale(nchannels) 
        self.relu = nn.ReLU(inplace=True)
        self.nchannels = nchannels

    def forward(self, x):
        x1 = self.scale(x)
        y = self.relu(x1)
        return y

    def __repr__(self):
        s = '{name} ({nchannels})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class PPReLU(nn.Module):
    def __init__(self, nchannels):
        super().__init__()
        self.scale1 = Scale(nchannels, bias=False, init_scale=1.0) 
        self.scale2 = Scale(nchannels, bias=False, init_scale=0.1) 
        self.nchannels = nchannels

    def forward(self, x):
        x1 = self.scale1(x)
        x2 = self.scale2(x)
        y = torch.max(x1, x2)
        return y

    def __repr__(self):
        s = '{name} ({nchannels})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class PLU(nn.Module):
    """
    y = max(alpha*(x+c)−c, min(alpha*(x−c)+c, x))
    from PLU: The Piecewise Linear Unit Activation Function
    """
    def __init__(self, alpha=0.1, c=1):
        super().__init__()
        self.alpha = alpha
        self.c = c

    def forward(self, x):
        x1 = self.alpha*(x + self.c) - self.c
        x2 = self.alpha*(x - self.c) + self.c
        min1 = torch.min(x2, x)
        min2 = torch.max(x1, min1)
        return min2

    def __repr__(self):
        s = '{name} ({alhpa}, {c})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class CReLU(nn.Module):
    def __init__(self, nchannels):
        super().__init__()
        self.scale = Scale(2*nchannels) 
        self.relu = nn.ReLU(inplace=True)
        self.in_channels = nchannels
        self.out_channels = 2*nchannels

    def forward(self, x):
        x1 = torch.cat((x, -x), 1)
        x2 = self.scale(x1)
        y = self.relu(x2)
        return y

    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class L2Norm(nn.Module):
    def __init__(self, nchannels, bias=True):
        super().__init__()
        self.scale = Scale(nchannels, bias=bias) 
        self.nchannels = nchannels
        self.eps = 1e-6

    def forward(self, x):
        #norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x = torch.div(x,norm)
        l2_norm = x.norm(2, dim=1, keepdim=True) + self.eps
        x_norm = x.div(l2_norm)
        y = self.scale(x_norm)
        return y

    def __repr__(self):
        s = '{name} ({nchannels})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class Conv2dL2NormLeaky(nn.Module):
    """ This convenience layer groups a 2D convolution, a batchnorm and a leaky ReLU.
    They are executed in a sequential manner.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the kernel of the convolution
        stride (int or tuple): Stride of the convolution
        padding (int or tuple): padding of the convolution
        leaky_slope (number, optional): Controls the angle of the negative slope of the leaky ReLU; Default **0.1**
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, leaky_slope=0.1, bias=True):
        super().__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if isinstance(kernel_size, (list, tuple)):
            self.padding = [int(ii/2) for ii in kernel_size]
        else:
            self.padding = int(kernel_size/2)
        self.leaky_slope = leaky_slope

        # Layer
        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False),
            L2Norm(self.out_channels, bias=bias),
            nn.LeakyReLU(self.leaky_slope, inplace=True)
        )

    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, negative_slope={leaky_slope})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
        x = self.layers(x)
        return x


## shufflenet
class Shuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        """
        Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]
        """
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C/g, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)

    def __repr__(self):
        s = '{name} (groups={groups})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

# mobilenet
class Conv2dBatchReLU(nn.Module):
    """ This convenience layer groups a 2D convolution, a batchnorm and a ReLU.
    They are executed in a sequential manner.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the kernel of the convolution
        stride (int or tuple): Stride of the convolution
        padding (int or tuple): padding of the convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, isPadding=True, isBias=False):
        super(Conv2dBatchReLU, self).__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.isBias = isBias
        if isinstance(kernel_size, (list, tuple)):
            self.padding = [int(ii/2) for ii in kernel_size]
        else:
            self.padding = int(kernel_size/2)

        # Layer
        if isPadding == True:

            self.layers = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=self.isBias),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True)
            )
        else:

            self.layers = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, 0, bias=self.isBias),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True)
            )


    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self,x):

        x = self.layers(x)
        # x = self.layer1(x)
        # x = self.layer2(x)

        return x

    def toCaffe(self):
        pass
        weightList = util.annalysis_pytorch_layer_name(self.layers, isBias=False)
        layerNameList = ["one"]

        return layerNameList, weightList

class layerConcat(nn.Module):
    def __init__(self, dim):
        super(layerConcat, self).__init__()
        self.dim = dim

    def forward(self, x):
        #print(type(x), x)
        return torch.cat(x, self.dim)

class Conv2dDepthWise(nn.Module):
    """ This layer implements the depthwise separable convolution from Mobilenets_.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the kernel of the convolution
        stride (int or tuple): Stride of the convolution
        padding (int or tuple): padding of the convolution

    .. _Mobilenets: https://arxiv.org/pdf/1704.04861.pdf
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Conv2dDepthWise, self).__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if isinstance(kernel_size, (list, tuple)):
            self.padding = [int(ii/2) for ii in kernel_size]
        else:
            self.padding = int(kernel_size/2)

        # Layer
        if in_channels != out_channels:
            self.layers = nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, self.kernel_size, self.stride, self.padding, groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),
                nn.ReLU(inplace=True),

                vn_layer.Conv2dBatchReLU(in_channels, out_channels, 1, 1),
            )
        else:
            self.layers = nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, self.kernel_size, self.stride, self.padding, groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),
                nn.ReLU(inplace=True),

                Conv2dBatchReLU(in_channels, out_channels, 1, 1),
            )

    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
        x = self.layers(x)
        return x

layrNum = 0

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, isBias=False):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.expand_ratio = expand_ratio
        self.use_res_connect = self.stride == 1 and inp == oup
        self.isBias = isBias
        #print("InvertedResidual in : ", inp, oup, expand_ratio)
        if abs(expand_ratio - 1) < .01:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=self.isBias),
                nn.BatchNorm2d(inp * expand_ratio),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=self.isBias),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=self.isBias),
                nn.BatchNorm2d(inp * expand_ratio),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=self.isBias),
                nn.BatchNorm2d(inp * expand_ratio),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=self.isBias),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        #print("in", x.size())
        if self.use_res_connect:
            out =  x + self.conv(x)
        else:
            out = self.conv(x)
       # print("************************** InvertedResidual weight begin")
        #if len(self.conv) == 5:
        #    print("***", self.conv[2])
        #else:
        #    print("****", self.conv[2], self.conv[5])
        global layrNum
       # print(layrNum)
        if layrNum == 16 and 0:
            #print(self.conv[0].weight)
            #print(self.conv[1].weight)
            #print(self.conv[1].bias)
            #print(self.conv[1].running_mean)
           # print(self.conv[1].running_var)
            #print(self.conv[2])
            if isinstance(self.conv[2], nn.ReLU):
                print("****************************relu")

            #print(self.conv[3].weight)
            #print(self.conv[4].weight)
            #print(self.conv[4].bias)
            #print(self.conv[4].running_mean)
            #print(self.conv[4].running_var)
#
            print("out", out)
        #sys.exit()
        layrNum += 1
        return out 

    def toCaffe(self):
        pass
        #layerNamesList = []
        #layerWeightsList = []
        #if self.use_res_connect:
       #     layerNamesList.append("twoAdd")

       #     weightList = util.annalysis_pytorch_layer_name(self.conv, self.isBias)
        #    return layerNamesList, weightList
       # else:
       #     layerNamesList.append("one")
#
        #    weightList = util.annalysis_pytorch_layer_name(self.conv, self.isBias)
        #    return layerNamesList, weightList

#class buildInvertedResBlock():
#    def __init__(residual_setting, input_channel):
#        self.t, self.c, self.n, self.s = 

def buildInvertedResBlock(residual_setting, input_channel):
    # building inverted residual blocks
    features = []
    t, c, n, s = residual_setting
    output_channel = int(c)
    layerNameList = []
    weightList = []
    #print("******************************************\n")
    for i in range(n):
        print("channel: ", input_channel, output_channel)
        if i == 0:
            curLayer = InvertedResidual(input_channel, output_channel, s, t)
            #features.append(InvertedResidual(input_channel, output_channel, s, t))
            features.append(curLayer)
          #  curLayerNameList, curweightList = curLayer.toCaffe()
        else:
            curLayer = InvertedResidual(input_channel, output_channel, 1, t)
            #features.append(InvertedResidual(input_channel, output_channel, 1, t))
            features.append(curLayer)
           # curLayerNameList, curweightList = curLayer.toCaffe()
        input_channel = output_channel
    layers = nn.Sequential(*features)
    return layers, output_channel


class ConvTranspose2dBatchReLU(nn.Module):
    """ This convenience layer groups a 2D convolution, a batchnorm and a ReLU.
    They are executed in a sequential manner.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the kernel of the convolution
        stride (int or tuple): Stride of the convolution
        padding (int or tuple): padding of the convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=1):
        super(Conv2dBatchReLU, self).__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding


        self.layers = nn.Sequential(
            nn.ConvTranspose2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True))


    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self,x):

        x = self.layers(x)

        return x

