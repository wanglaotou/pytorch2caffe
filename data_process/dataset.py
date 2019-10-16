'''
@Author: Jiangtao
@Date: 2019-08-07 10:42:06
@LastEditors: Jiangtao
@LastEditTime: 2019-09-23 11:36:08
@Description: 
'''

import torch
from torch.utils.data.dataset import Dataset as torchDataset
import sys
import cv2
import numpy as np
import random
import time

__all__ = ['Dataset', 'DatasetWithAngle', 'DatasetWithAngleMulti2']


def get_points(line):

    pts = line[1:]
    pts = np.array(pts)
    pts = pts.reshape((8,))

    return pts
    
def read_image_points(path):

    with open(path, "r") as file:

        imgPath = []
        points = []

        for line in file.readlines():

            line = line.strip().split(' ')

            points_ = get_points(line)

            imgPath.append(line[0])
            points.append(points_)

    return imgPath, points
def sample_image_points(pathList):

    imagepathList = []
    pointsList = []

    shuffleList = []

    newImagepathList = []
    newPointsList = []

    for i in range(len(pathList)):
        curImagepathList, curPointsList = read_image_points(pathList[i])

        imagepathList.extend(curImagepathList)
        pointsList.extend(curPointsList)

    if len(imagepathList) != len(pointsList):
        print("image_smaple has some problem!\n")
        sys.exit()

    for i in range(len(imagepathList)):
        curList = []
        curList.append(imagepathList[i])
        curList.append(pointsList[i])

        shuffleList.append(curList)

    random.shuffle(shuffleList)

    for i in range(len(shuffleList)):

        newImagepathList.append(shuffleList[i][0])
        newPointsList.append(shuffleList[i][1])

    return newImagepathList, newPointsList

class DatasetWithAngleMulti3(torchDataset):

    def __init__(self, imglistPath, inputSize, img_tf, label_tf, imgChannel=1,isTrain='train'):

        if isinstance(imglistPath, (list, tuple)):
            self.imgPathList, self.eyeList = sample_image_points(imglistPath)
        else:
            self.imgPathList, self.eyeList = read_image_points(imglistPath)
        # print(self.imgPathList)
        # print(self.labelList)
        if isTrain == 'train':

            nTrain = int(len(self.imgPathList) * 0.8)

            self.imgPathList = self.imgPathList[0:nTrain]
            self.eyeList = self.eyeList[0:nTrain]

        if isTrain == 'val':

            nTrain = int(len(self.imgPathList) * 0.8)

            self.imgPathList = self.imgPathList[nTrain:]
            self.eyeList = self.eyeList[nTrain:]


        if isTrain == 'trainval':

            self.imgPathList = self.imgPathList
            self.eyeList = self.eyeList

        self.img_tf = img_tf
        self.label_tf = label_tf
        self.num = 0
        self.channel = imgChannel

    def __len__(self):

        return len(self.imgPathList)

    def __getitem__(self, index):

        if(self.channel == 1):

            img = cv2.imread("/home/mario/Projects/LPR/License_Plate_Detection_Pytorch/MTCNN/data_preprocessing/utilisation_license/" + self.imgPathList[index], 0)

        else:
            img = cv2.imread("/home/mario/Projects/LPR/License_Plate_Detection_Pytorch/MTCNN/data_preprocessing/utilisation_license/" + self.imgPathList[index])

        if not isinstance(img, np.ndarray) or img.shape[0] <= 0 or img.shape[1] <= 0:
            print("img none! and is {}\n".format(self.imgPathList[index]))
            sys.exit()

        if self.img_tf is not None:
            # print(type(img))
            # print(img.shape)
            # img = img[np.newaxis,:,:]
            # print(img.shape)
            img = self.img_tf(img)

                
        return img, self.eyeList[index]

    def collate_fn(self, batch):
        
        images = list()
        eye = list()


        for b in batch:

            images.append(b[0].float())
            eye.append(b[1].tolist())
       
        eye = np.array(eye,dtype=np.float64)

        images = torch.stack(images, dim=0)
        images = torch.FloatTensor(images)
        eye = torch.FloatTensor(eye)

        return images, eye