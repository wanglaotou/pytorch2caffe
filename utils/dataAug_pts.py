'''
@Author: Jiangtao
@Date: 2019-07-24 11:31:08
@LastEditors: Jiangtao
@LastEditTime: 2019-09-21 17:33:13
@Description: 
'''
import copy
import math
import os
import pickle
import random
import time

import numpy as np

import cv2

# 根据box进行合理放大
def randomeResize(img,pts):

    height ,width = img.shape[0:2]

    minX,minY = pts.min(axis=0)
    maxX,maxY = pts.max(axis=0)

    w = maxX - minX
    h = maxY - minY

    delta_x1 = np.random.randint(int(w * 0), int(w * 0.5))
    delta_y1 = np.random.randint(int(h * 0), int(h * 0.5))
    # delta_x2 = np.random.randint(int(w * 0), int(w * 0.5)) 
    # delta_y2 = np.random.randint(int(h * 0), int(h * 0.5))
    # print(delta_x1, delta_y1)

    nx1 = max(minX - delta_x1,0)
    ny1 = max(minY - delta_y1,0)
    nx2 = min(maxX + delta_x1,width)
    ny2 = min(maxY + delta_y1,height)
    
    # print('in before:', pts[0], pts[1])
    pts[:,0] -= delta_x1
    pts[:,1] -= delta_y1
    # print('in after:', pts[0], pts[1])

    if len(img.shape) >2:
        img = img[int(ny1): int(ny2), int(nx1): int(nx2), :]
    if len(img.shape) == 2:
        img = img[int(ny1): int(ny2), int(nx1): int(nx2)]

    return img,pts

# 随机裁剪
def randomeCrop(img,pts):

    height ,width = img.shape[0:2]

    minX,minY = pts.min(axis=0)
    maxX,maxY = pts.max(axis=0)

    x1,y1,x2,y2 = minX,minY,maxX,maxY

    w = x2 - x1
    h = y2 - y1


    if w*0.5 <=0 or h*0.5<=0:
        return img,pts
        
    flag =  random.random()

    if flag >= 0.5:
        delta_x1 = np.random.randint(0,int(w * 0.5))
        delta_y1 = np.random.randint(0,int(h * 0.5))
        # delta_x2 = np.random.randint(0,int(w * 0.5)) 
        # delta_y2 = np.random.randint(0,int(h * 0.5)) 
    else:
        delta_x1 = np.random.randint(int(w * 0.5), int(w * 1))
        delta_y1 = np.random.randint(int(h * 0.5), int(h * 1))
        # delta_x2 = np.random.randint(int(w * 0.5), int(w * 1)) 
        # delta_y2 = np.random.randint(int(h * 0.5), int(h * 1))


    nx1 = max(x1 - delta_x1,0)
    ny1 = max(y1 - delta_y1,0)
    nx2 = min(x2 + delta_x1,width)
    ny2 = min(y2 + delta_y1,height)
    
    pts[:,0] -= delta_x1
    pts[:,1] -= delta_y1

    if len(img.shape) >2:
        img = img[int(ny1): int(ny2), int(nx1): int(nx2), :]
    if len(img.shape) == 2:
        img = img[int(ny1): int(ny2), int(nx1): int(nx2)]

    return img,pts

# 随机旋转+-15度
def randomeRotate(img,pts):

    num = 15
    angle = np.random.randint(-num,num)
    rad = angle * np.pi / 180.0

    height ,width = img.shape[0:2]
    center = (width / 2, height / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (width, height))

    pts_ = np.zeros_like(pts)
    pts[:,1] = height - pts[:,1]
    pts_[:,0] = (pts[:,0] - (width / 2)) * np.cos(rad) - (pts[:,1] - (height / 2)) * np.sin(rad) + (width / 2)
    pts_[:,1] = (pts[:,1] - (height / 2)) * np.cos(rad) + (pts[:,0] - (width / 2)) * np.sin(rad) + (height / 2)
    pts_[:,1] = height - pts_[:,1]
    
    return rotated,pts_

# 随机对称
def randomFlip(img,pts):

    height , width = img.shape[0:2]
    print('height,width:',height,width)
    flipped_img = cv2.flip(img,1)
    pts_ = np.zeros_like(pts)
    pts_[:,1] = pts[:,1]
    pts_[:,0] = width - pts[:,0]
    # pts = pts_

    # for i,j in [[0,16],[36,45],[37,44],[41,46],[39,42],[48,54]]:
    #     temp = copy.deepcopy(pts[i])
    #     pts[i] = pts[j]
    #     pts[j] = temp

    # for i,j in [[0,1],[2,3]]:
    #     temp = copy.deepcopy(pts[i])
    #     pts[i] = pts[j]
    #     pts[j] = temp

    return flipped_img,pts_

# 随机偏移
def randomTranslation(img,pts):

    height,width = img.shape[0:2]
    minX,minY = pts.min(axis=0)
    maxX,maxY = pts.max(axis=0)
    w = min(minX,width - maxX)
    h = min(minY,height - maxY)
    # print(w,h)
    if random.choice([True,False]):
        w = -w
    if random.choice([True,False]):
        h = -h
    if w > 0:          
        w = random.randint(0, int(w))
    else:               
        w = random.randint(int(w), 0)               
    if h > 0:
        h = random.randint(0, int(h))         
    else:
        h = random.randint(int(h), 0)

    affine = np.float32([[1,0,w],[0,1,h]])
    img = cv2.warpAffine(img,affine,(img.shape[1],img.shape[0]))
    pts += np.array([w,h])

    return img,pts

# 随机HSV空间变化
def randomHSV(img,pts):

    hue_vari = 10
    sat_vari = 0.2
    val_vari = 0.2

    hue_delta = np.random.randint(-hue_vari, hue_vari)
    sat_mult = 1 + np.random.uniform(-sat_vari, sat_vari)
    val_mult = 1 + np.random.uniform(-val_vari, val_vari)

    if len(img.shape) == 2:
        img = cv2.merge([img,img,img])

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float)
    img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
    img_hsv[:, :, 1] *= sat_mult
    img_hsv[:, :, 2] *= val_mult
    img_hsv[img_hsv > 255] = 255
    img = cv2.cvtColor(np.round(img_hsv).astype(np.uint8), cv2.COLOR_HSV2BGR)

    return img,pts

# 随机滤波
def randomBlur(img,pts):

    img_mean = cv2.blur(img, (5,5))
    img_Guassian = cv2.GaussianBlur(img,(5,5),0)
    img_median = cv2.medianBlur(img, 5)
    img_bilater = cv2.bilateralFilter(img,9,75,75)
    
    n = random.random()
    if n<=0.25:
        return img_mean,pts
    if n <= 0.5:
        return img_Guassian,pts
    if n <= 0.75:
        return img_median,pts
    if n <= 1:
        return img_bilater,pts

# 随机噪声
def randomNoise(img,pts):

    N = int(img.shape[0] * img.shape[1] * 0.001)
    for i in range(N): #添加点噪声
        temp_x = np.random.randint(0,img.shape[0])
        temp_y = np.random.randint(0,img.shape[1])
        img[temp_x][temp_y] = 255

    return img,pts

# 集成随机
def randomAug(img,pts):

    # if random.random() > 0.5:
    #     img,pts = randomeCrop(img,pts)
    # img,pts = randomeResize(img,pts)
    # print('resize after:', img.shape)

    if random.random() > 0.5:
        img,pts = randomeRotate(img,pts)

    # if random.random() > 0.5:
    #     img,pts = randomFlip(img,pts)

    # if random.random() > 0.5:
    #     img,pts = randomTranslation(img,pts)
    
    if random.random() > 0.5:
        img,pts = randomHSV(img,pts)

    if random.random() > 0.5:
        img,pts = randomBlur(img,pts)

    if random.random() > 0.5:
        img,pts = randomNoise(img,pts)
    
    img,pts = randomeCrop(img,pts)

    return img,pts


if __name__ == '__main__':
    


    while True:

        imgFile = os.path.join('./img0_1/','0000000000000000-180530-150700-151025-000006000400_320.jpg')
        ptsFile = os.path.join('./pts0_1/','0000000000000000-180530-150700-151025-000006000400_320.pts')

        img = cv2.imread(imgFile,-1)
        pts = np.genfromtxt(ptsFile,skip_header=3,skip_footer=1)

        img,pts = randomAug(img,pts)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in [0,16,36,45,37,44,41,46,39,42,48,54]:

            cv2.circle(img,(int(pts[i][0]),int(pts[i][1])),3,(255,0,0),-1)
            cv2.putText(img,str(i),(int(pts[i][0]),int(pts[i][1])), font, 0.4, (255, 255, 255), 1)
        cv2.imshow('img',img)
        cv2.waitKey(0)

        