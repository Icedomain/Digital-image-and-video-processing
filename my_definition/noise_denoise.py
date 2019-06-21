"""
220180776 胡欣毅
王桥　数字图像处理　自编函数库

"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage


# 加噪
'''
noise_img = skimage.util.random_noise(img, mode="gaussian")
'''

def add_noise(img):
    # 调整幅度
    Amplitude = 255
    mean, var = 0, 0.01
    noise = np.random.normal(mean, var ** 0.5, img.shape)
    img = img.astype(float)
    out = abs(img + noise * Amplitude)
    return out.astype(int)


# 椒盐噪点
def salt(img, n):
    for k in range(n):
        i = int(np.random.random() * img.shape[1])
        j = int(np.random.random() * img.shape[0])
        if len(img.shape) == 2:
            img[j,i] = 255
        elif len(img.shape) == 3:
            img[j,i,:]= 255,255,255
        return img


# 加高斯噪声
def gaussian_noise(src):
    out = np.zeros_like(src)
    for row in range(src.shape[0]):
        for col in range(src.shape[1]):
            if len(src.shape) == 2: 
                s = np.random.normal(0, 20)
                # clip 限幅
                out[row, col] = np.clip(src[row, col] + s ,0,255)
            elif len(src.shape) == 3: 
                s = np.random.normal(0, 20, 3)
                # clip 限幅
                out[row, col, :] = np.clip(src[row, col, :] + s ,0,255)   
    return out




#　滤波
# 中值滤波
'''
denoise = cv2.medianBlur(img, ksize=3)
'''
# 均值滤波
def mean_denoise(src ,ksize = (3,3) ):
    ker = np.ones(ksize)/(ksize[0]*ksize[1])
    out = cv2.filter2D(src, -1 , ker )
    return out


# 均值滤波
'''
denoise = cv2.medianBlur(img , 3) 
'''
# 计算中值滤波
def median_denoise(src , ksize):
    # 输出
    outputMap = np.zeros_like(src)
    for i in range(src.shape[0]):
        for j in range(src.shape[1]): 
            # 计算中值
            h_begin = max(0, i - ksize // 2)
            h_end = min(src.shape[0], i + ksize // 2)
            w_begin = max(0, j - ksize // 2)
            w_end = min(src.shape[1], j + ksize // 2)
    
            if len(src.shape) == 2:  
                outputMap[i, j] = np.median(src[h_begin:h_end, w_begin:w_end])
            elif len(src.shape) == 3:
                outputMap[i, j,:] = np.median(src[h_begin:h_end, w_begin:w_end, :]) 
    return outputMap




# 高斯滤波
'''
denoise = cv2.GaussianBlur(img, ksize=3)
'''



# 双边滤波 
'''
denoise = cv2.bilateralFilter(img,9,75,75)
'''



















