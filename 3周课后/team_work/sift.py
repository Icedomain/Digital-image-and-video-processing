

# https://blog.csdn.net/zhangziju/article/details/79754652

import cv2
import numpy as np
from matplotlib import pyplot as plt

#   参数设定

image = "../data/4.1.05.tiff"

model = cv2.xfeatures2d.SIFT_create()
# model = cv2.xfeatures2d.SURF_create()

def rotate(image, angle, center=None, scale=1.0):
    ##　h,w 尺寸
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
 
    M = cv2.getRotationMatrix2D(center, angle, scale)
    # print(M) ## 旋转阵　＋　旋转中心
    # 旋转
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated



img1 = cv2.imread(image)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
kp1, des1 = model.detectAndCompute(img1,None) 
#des是描述子

img2  = rotate(img1, 45,scale = .5 ) 
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
kp2, des2 = model.detectAndCompute(img2,None)

#水平拼接
together = np.hstack((img1, img2)) 
cv2.imshow('src',together )

together = np.hstack((gray1, gray2)) 
cv2.imshow('gray',together )

img3 = cv2.drawKeypoints(img1,kp1,img1,color=(255,0,255))
img4 = cv2.drawKeypoints(img2,kp2,img2,color=(255,0,255))

together = np.hstack((img3, img4)) 
cv2.imshow("point", together)


# FLANN 参数设计
# 快速最近邻搜索包（Fast_Library_for_Approximate_Nearest_Neighbors）
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)
# print(matches)
matchesMask = [[0,0] for i in range(len(matches)) ]

good = []
for m,n in matches:
    if m.distance < 0.7 * n.distance:
        good.append([m])
print(good)

img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,flags=2)
cv2.imshow("FLANN matches", img5)
img6 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
cv2.imshow("FLANN matches", img6)


cv2.waitKey (0)  
cv2.destroyAllWindows() 



