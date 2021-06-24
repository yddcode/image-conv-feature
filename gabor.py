import cv2
import numpy as np
import os
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
    contrast = greycoprops(glcm,'contrast')
    dissimilarity = greycoprops(glcm,'dissimilarity')
    homogeneity = greycoprops(glcm, 'homogeneity')
    correlation = greycoprops(glcm, 'correlation')
    # temp = greycoprops(glcm, 'energy')
    # print(temp)
    ASM = greycoprops(glcm, 'ASM')
    # temp1 = np.array(temp)
    # temp2 = list(chain.from_iterable(temp1))
    return contrast, dissimilarity, homogeneity, ASM

#构建Gabor滤波器
def build_filters():
     filters = []
     ksize = [9,10,11,12,13] # gabor尺度，6个
     lamda = np.pi/6.0         # 波长
     for theta in np.arange(0, np.pi, np.pi / 8): #gabor方向，0°，45°，90°，135°，共四个
         for K in range(5):
             kern = cv2.getGaborKernel((ksize[K], ksize[K]), 7.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
             kern /= 1.5*kern.sum()
             filters.append(kern)
     return filters


if __name__ == '__main__':
    filters = build_filters()
    print(len(filters))
    a = 0 # extraction\纹理特征\Gabor\1.jpg
    contrast = np.zeros((40000,40), dtype=np.float32)
    dissimilarity = np.zeros((40000,40), dtype=np.float32)
    homogeneity = np.zeros((40000,40), dtype=np.float32)
    ASM = np.zeros((40000,40), dtype=np.float32)

    for filename in os.listdir('D:/guji_resizedata510/'):
    # filename = './extraction/wenli/Gabor/1.jpg'
        img = cv2.imread('D:/guji_resizedata510/' + filename, 0)
        print(filename)
        for i in range(len(filters)):
            accum = np.zeros_like(img)
            for kern in filters[i]:
                fimg = cv2.filter2D(img, cv2.CV_8UC1, kern)
                accum = np.maximum(accum, fimg, accum)
                # print(i)
                # cv2.imwrite(str(i) + ".jpg", accum)
            contrast1, dissimilarity1, homogeneity1, ASM1 = get_inputs(accum)
            # print(contrast, np.mean(contrast))
            # energy.append(np.mean(temp))
            # for i in range(40):
            ASM[a, i] = np.mean(ASM1)
            contrast[a, i] = np.mean(contrast1)
            dissimilarity[a, i] = np.mean(dissimilarity1)
            homogeneity[a, i] = np.mean(homogeneity1)
            
        a += 1    
    contrast = np.array(contrast)
    dissimilarity = np.array(dissimilarity)
    homogeneity = np.array(homogeneity)
    ASM = np.array(ASM)
    # energy = list(chain.from_iterable(energy))
    # print('mena:', mean)
    contrast = pd.DataFrame(data=contrast)
    contrast.to_csv("./extraction/contrast.csv", encoding='utf-8', index=False)
    dissimilarity = pd.DataFrame(data=dissimilarity)
    dissimilarity.to_csv("./extraction/dissimilarity.csv", encoding='utf-8', index=False)
    homogeneity = pd.DataFrame(data=homogeneity)
    homogeneity.to_csv("./extraction/homogeneity.csv", encoding='utf-8', index=False)
    ASM = pd.DataFrame(data=ASM)
    ASM.to_csv("./extraction/ASM.csv", encoding='utf-8', index=False)
        # a += 1
        # if a == 1:
        #     break
