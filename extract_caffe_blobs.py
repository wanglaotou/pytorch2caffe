import math
import numpy as np
import os
import sys
import cv2
import caffe

np.set_printoptions(threshold=sys.maxsize)

file_path = "./12/2858.jpg"
PNet_model_def = "pnet.prototxt"
PNet_model_weights = "pnet.caffemodel"

# caffe.set_device(1)
caffe.set_mode_cpu()

# Load models.
PNet = caffe.Net(PNet_model_def, PNet_model_weights, caffe.TEST)

# Transform to fill data.
im = cv2.imread(file_path, 1)
if im.shape[2] == 1:
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
im = im.astype(np.float32)
print('Image In:', im.shape, 'Net In:',PNet.blobs['data'].data.shape)

# bgr -> rgb
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) #im = im[...,::-1]

im_resized = cv2.resize(im, (PNet.blobs['data'].data.shape[3], PNet.blobs['data'].data.shape[2]), 0, 0, interpolation=cv2.INTER_LINEAR)
# h,w,c -> c,h,w
im_resized = np.transpose(im_resized, (2, 0, 1))  
im_resized = (im_resized - 127.5) * 0.0078125
# c,h,w -> 1,c,h,w
PNet.blobs['data'].data[0] = im_resized	
'''
im = cv2.resize(im, (PNet.blobs['data'].data.shape[3], PNet.blobs['data'].data.shape[2]), 0, 0, interpolation=cv2.INTER_LINEAR)
transformer = caffe.io.Transformer({'data': PNet.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1)) 
transformer.set_mean('data', np.array((127.5, 127.5, 127.5)))  
transformer.set_raw_scale('data', 1/127.5)
im = transformer.preprocess('data', im)
PNet.blobs['data'].data[...] = im	
'''
# Extract the net output blobs.
outputs = PNet.forward()
for blob in outputs.keys():
	fn = "./" + blob + ".txt"
	outf = open(fn, "w")
	outf.write(str(outputs[blob]))
	outf.close()
print(outputs.keys())
