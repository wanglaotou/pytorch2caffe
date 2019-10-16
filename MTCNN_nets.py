import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

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

        # def conv_dw(inp, oup, stride, kernel_size):
        #     return nn.Sequential(
        #         nn.Conv2d(inp, inp, kernel_size, stride, 0, groups=inp, bias=True),
        #         nn.BatchNorm2d(inp),
        #         nn.ReLU(inplace=True),
    
        #         nn.Conv2d(inp, oup, 1, 1, 0, bias=True),
        #         nn.BatchNorm2d(oup),
        #         nn.ReLU(inplace=True),
        #     )

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 10, 3, 1)),
            # ('conv1', conv_dw(3, 10, 1, (3, 3))),
            ('prelu1', nn.PReLU(10)),
            ('pool1', nn.MaxPool2d((3,5), ceil_mode=True)),     

            ('conv2', nn.Conv2d(10, 16, (3,5), 1)),
            # ('conv2', conv_dw(10, 16, 1, (3, 5))),
            ('prelu2', nn.PReLU(16)),

            ('conv3', nn.Conv2d(16, 32, (3,5), 1)),
            # ('conv3', conv_dw(16, 32, 1, (3,5))),
            ('prelu3', nn.PReLU(32))
        ]))

        self.conv4_1 = nn.Conv2d(32, 2, 1, 1)
        # self.conv4_2 = nn.Conv2d(32, 8, 1, 1)
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
        ## depthwise conv
        # def conv_dw(inp, oup, stride, kernel_size):
        #     return nn.Sequential(
        #         nn.Conv2d(inp, inp, kernel_size, stride, 0, groups=inp, bias=True),
        #         nn.BatchNorm2d(inp),
        #         nn.ReLU(inplace=True),
    
        #         nn.Conv2d(inp, oup, 1, 1, 0, bias=True),
        #         nn.BatchNorm2d(oup),
        #         nn.ReLU(inplace=True),
        #     )

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 32, 3, 1)),
            # ('conv1', conv_dw(3, 32, 1, (3, 3))),
            ('prelu1', nn.PReLU(32)),
            ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),
            # ('pool1', nn.Conv2d(32, 32, 3, 2, 1)),

            ('conv2', nn.Conv2d(32, 64, (5,3), 1)),
            # ('conv2', conv_dw(32, 64, 1, (3, 3))),
            ('prelu2', nn.PReLU(64)),
            ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv3', nn.Conv2d(64, 64, (5,3), 1)),
            # ('conv3', conv_dw(64, 64, 1, (3, 3))),
            ('prelu3', nn.PReLU(64)),
            ('pool3', nn.MaxPool2d(2, 2, ceil_mode=True)),

            ('conv4', nn.Conv2d(64, 128, 1, 1)),
            # ('conv4', conv_dw(64, 128, 1, (1, 1))),
            ('prelu4', nn.PReLU(128)),

            ('flatten', Flatten()),
            ('conv5', nn.Linear(1280, 256)),      #mario.c3
            ('drop5', nn.Dropout(0.25)),
            ('prelu5', nn.PReLU(256)),
        ]))

        self.conv6_1 = nn.Linear(256, 2)
        self.conv6_2 = nn.Linear(256, 4)
        # self.conv6_3 = nn.Linear(256, 8)

    def forward(self, x):
        x = self.features(x)
        a = self.conv6_1(x)
        b = self.conv6_2(x)
        # c = self.conv6_3(x)

        if self.is_train is False:
            a = F.softmax(a, dim=1)

        return b, a

    
if __name__ == "__main__":
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Pnet = PNet().to(device)
    Onet = ONet().to(device)
    
    P_input = torch.Tensor(2, 3, 17, 47).to(device)
    P_offset, P_prob = Pnet(P_input)
    print('P_offset shape is', P_offset.shape)
    print('P_prob shape is', P_prob.shape)
    
    O_input = torch.Tensor(2, 3, 34, 94).to(device)
    O_offset, O_prob = Onet(O_input)
    print('O_offset shape is', O_offset.shape)
    print('O_prob shape is', O_prob.shape)

    
    from torchsummary import summary
    summary(Pnet, (3,12,47))
    summary(Onet, (3,24,94))


