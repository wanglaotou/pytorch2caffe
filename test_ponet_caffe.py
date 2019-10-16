import math
import numpy as np
import os
import sys
import cv2
import caffe

np.set_printoptions(threshold=sys.maxsize)

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
# import dlib
import os
from imutils import paths
from collections import OrderedDict
import util_tf
import data_process

PNet_model_def = "pnet.prototxt"
PNet_model_weights = "pnet.caffemodel"
ONet_model_def = "onet.prototxt"
ONet_model_weights = "onet.caffemodel"
lmark_model_def = "landmark.prototxt"
lmark_model_weights = "landmark.caffemodel"

# caffe.set_device(1)
caffe.set_mode_cpu()

# Load models.
pnet = caffe.Net(PNet_model_def, PNet_model_weights, caffe.TEST)
onet = caffe.Net(ONet_model_def, ONet_model_weights, caffe.TEST)
landmark = caffe.Net(lmark_model_def, lmark_model_weights, caffe.TEST)

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

    img_paths = []
    img_paths += [el for el in paths.list_images(args.test_image)]
    random.shuffle(img_paths)
    num = len(img_paths)
    print("%d pics in total" % num)
    idx = 0 
    thresholds = 0.6, 0.7

     ## landmark
    for annotation in img_paths:
        im_path = annotation
        img_ori = cv2.imread(im_path)
        height, width, channel = img_ori.shape
        img_ori = cv2.cvtColor(img_ori, cv2.COLOR_RGB2GRAY)
        img_ori = img_ori.reshape((width, height, 1))

        if img_ori is None:
            print("---------img is empty---------",im_path)
            continue
        
        img = img_ori * 0.0039216
        # img = (img_ori - 127.5) * 0.0078125

        transformer = caffe.io.Transformer({'data': landmark.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        t1 = time.time()       
        out = landmark.forward_all(data=np.asarray([transformer.preprocess('data', img)]))
        print(time.time() - t1)
        # result = out
        landmarkout = out['fc1']

        print(landmarkout)
'''
    ## pnet, onet
    for annotation in img_paths:
        im_path = annotation

        im = cv2.imread(im_path)

        if im.shape[2] == 1:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        im = im.astype(np.float32)
        print('Image In:', im.shape, 'Net In:',onet.blobs['data'].data.shape)

        # bgr -> rgb
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) #im = im[...,::-1]
        
        ## pnet
        # im_resized = cv2.resize(im, (pnet.blobs['data'].data.shape[3], pnet.blobs['data'].data.shape[2]), 0, 0, interpolation=cv2.INTER_LINEAR)
        ## onet
        im_resized = cv2.resize(im, (onet.blobs['data'].data.shape[3], onet.blobs['data'].data.shape[2]), 0, 0, interpolation=cv2.INTER_LINEAR)
        # ## landmark
        # resize_size = (landmark.blobs['data'].data.shape[3], landmark.blobs['data'].data.shape[2])
        # im_resized = cv2.resize(im, resize_size, 0, 0, interpolation=cv2.INTER_LINEAR)
        # h,w,c -> c,h,w
        im_resized = np.transpose(im_resized, (2, 0, 1))  
        im_resized = (im_resized - 127.5) * 0.0078125
        
        # ## pnet
        # pnet.blobs['data'].data[0] = im_resized
        # outputs = pnet.forward()
        # for blob in outputs.keys():
        #     param = blob
        #     offset = outputs[blob]
        #     print(param, offset)

        # for blob in outputs.keys():
        #     fn = "./" + blob + ".txt"
        #     outf = open(fn, "w")
        #     outf.write(str(outputs[blob]))
        #     outf.close()
        # print(outputs.keys())

        ## onet
        onet.blobs['data'].data[0] = im_resized
        outputs = onet.forward()
        for blob in outputs.keys():
            param = blob
            offset = outputs[blob]
            print(param, offset)
        
        # ## landmark
        # landmark.blobs['data'].data[0] = im_resized
        # outputs = landmark.forward()
        # for blob in outputs.keys():
        #     param = blob
        #     offset = outputs[blob]
        #     print(param, offset)
'''
   