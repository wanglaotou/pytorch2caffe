'''
@Author: Jiangtao
@Date: 2019-08-07 10:42:06
@LastEditors: Jiangtao
@LastEditTime: 2019-09-23 15:09:21
@Description: 
'''

import torch
import cv2
import sys
import argparse
import numpy as np
from MTCNN_nets import PNet, ONet
from torchvision import transforms as tf
import random
import time
from utils.util import *
import os
from imutils import paths
from collections import OrderedDict
import util_tf
import data_process

__all__ = ['TestEngineImg']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

p_model_path='models/pnet_Weights' 
o_model_path='models/onet_Weights'
l_model_path='models/landmark.pkl'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pnet = PNet().to(device)
pnet.load_state_dict(torch.load(p_model_path, map_location=lambda storage, loc: storage))
pnet.eval()

onet = ONet().to(device)
onet.load_state_dict(torch.load(o_model_path, map_location=lambda storage, loc: storage))
onet.eval()

landmark = torch.load(l_model_path)
landmark = landmark.to(device)
landmark.eval()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MTCNN Demo')
    parser.add_argument("--test_image", dest='test_image', help=
    "test image path", default="./lmark", type=str)
    parser.add_argument("--scale", dest='scale', help=
    "scale the iamge", default=1, type=int)
    parser.add_argument('--mini_lp', dest='mini_lp', help=
    "Minimum lp to be detected. derease to increase accuracy. Increase to increase speed",
                        default=50, type=int)
                        # default=(50, 15), type=int)

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_paths = []
    img_paths += [el for el in paths.list_images(args.test_image)]
    random.shuffle(img_paths)
    num = len(img_paths)
    print("%d pics in total" % num)
    idx = 0 
    thresholds = 0.6, 0.7

    for annotation in img_paths:
        im_path = annotation
        img = cv2.imread(im_path, 0)
        print(im_path)
        draw = img.copy()
        ## pnet, onet
        # img = torch.FloatTensor(preprocess(img)).to(device)
        ## landmark
        rs  = data_process.transform.inputResize(img.shape)
        it  = tf.ToTensor()

        img_tf = util_tf.Compose([rs, it])
        data = img_tf(img)
        data = torch.unsqueeze(data, 0)
        data = data.to(device)
        # ## pnet
        # offset, prob = onet(img)
        # probs = prob.cpu().data.numpy()[0, 1, :, :]  # probs: probability of a face at each sliding window
        # offsets = offset.cpu().data.numpy() 
        # print(offsets,offsets.shape)
        
        # ## onet
        # offset, prob = onet(img)
        # offsets = offset.cpu().data.numpy()  # shape [n_boxes, 4]
        # probs = prob.cpu().data.numpy()
        # print(offsets,offsets.shape)

        ## landmark
        offset = landmark(data)  
        offsets = offset[0].cpu().detach().numpy()  # shape [landmark, 8]
        print(offsets,offsets.shape)
