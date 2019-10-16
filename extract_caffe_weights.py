#-*- coding: UTF-8 -*-
#!/usr/bin/env python  
import caffe  
import numpy as np  
import sys

# 使输出的参数完全显示
# 若没有这一句，因为参数太多，中间会以省略号“……”的形式代替
np.set_printoptions(threshold=sys.maxsize)
  
# deploy文件
MODEL_FILE = './det3.prototxt'
# 预先训练好的caffe模型
PRETRAIN_FILE = './det3.caffemodel'
  
# 保存参数的文件
params_txt = './params.txt'
pf = open(params_txt, 'w')  
  
# 让caffe以测试模式读取网络参数
net = caffe.Net(MODEL_FILE, PRETRAIN_FILE, caffe.TEST)

for layer_name, blob in net.blobs.iteritems():
    print(layer_name + '\t' + str(blob.data.shape))

# 遍历每一层
for param_name in net.params.keys():
    # 该层在prototxt文件中对应“top”的名称
    pf.write(param_name)  
    pf.write('\n')  

    for i in range(len(net.params[param_name])):
        pf.write('\n' + param_name + '_data[' + str(i) + ']:\n\n')
        weight = net.params[param_name][i].data
        pf.write(str(weight))
  
    pf.write('\n\n')  
  
pf.close

