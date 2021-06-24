"""
提取glcm纹理特征
"""
import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops
import operator
from functools import reduce
from itertools import chain

def get_inputs(input):  # s为图像路径
    # 得到共生矩阵，参数：图像矩阵，距离，方向，灰度级别，是否对称，是否标准化
    # glcm = greycomatrix(input, [16, 32, 48, 64, 80, 96, 112, 128, 144, 160], [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4], 256, symmetric=True, normed=True)
    glcm = greycomatrix(input, [1], [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4], 256, symmetric=True, normed=True)
    # 得到共生矩阵统计值，http://tonysyu.github.io/scikit-image/api/skimage.feature.html#skimage.feature.greycoprops
    #temp = greycoprops(glcm,'contrast')
    #temp = greycoprops(glcm,'dissimilarity')
    #temp = greycoprops(glcm, 'homogeneity')
    #temp = greycoprops(glcm, 'correlation')
    temp = greycoprops(glcm, 'energy')
    print(temp)
    #temp = greycoprops(glcm, 'ASM')
    # temp1 = np.array(temp)
    # temp2 = list(chain.from_iterable(temp1))
    return temp

if __name__ == '__main__':
    energy = []
    energys = []
    i = 0
    mean = np.zeros((1,40), dtype=np.float32)
    for filename in os.listdir('./extraction/Image/'):
        
        if filename.endswith('jpg'):
            print(filename)
            img = cv2.imread('./extraction/Image/' + filename, cv2.IMREAD_GRAYSCALE)
            temp = get_inputs(img)
            print(temp, np.mean(temp))
            # energy.append(np.mean(temp))
            # for i in range(40):
            mean[0, i] = np.mean(temp)
            i += 1

    mean = np.array(mean)
    # energy = list(chain.from_iterable(energy))
    print('mena:', mean)
    mean = pd.DataFrame(data=mean)
    mean.to_csv("./extraction/energy.csv", encoding='utf-8', index=False)
            # f = open('C:\\Users\Administrator\\PycharmProjects\\untitled9\\lan\\' + "energy.txt", 'a')
            # f.write(str(temp))
            # f.write('\r\n')
