"""
220180776 胡欣毅
王桥　数字图像处理　自编函数库

"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

def show(src , cmap = 'gray'):
    # plt.figure()
    plt.imshow(src, cmap = cmap )
    plt.axis("off")
    plt.show()

def read_rgb(src):
    im = cv2.imread(src)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    print(im.shape)
    show(im)
    return im

def read_gray(src):
    im = cv2.imread(src)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # gray = rgb2gray(im)
    print(gray.shape)
    show(gray)
    return gray

# A - B 之间的拉倒255
def setAB(A,B,image):
    image = np.array((image >= A) * (image <=B) , dtype='int') *255
    return image

# 旋转
def rotate(image, angle, center=None, scale=1.0 , bound = False):
    ##　h,w 尺寸
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
 
    M = cv2.getRotationMatrix2D(center, angle, scale)
    # print(M) ## 旋转阵　＋　旋转中心

    # 旋转 bound == False时不填充，边缘有裁减
    if bound == False:
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated
    elif bound == True:
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - center[0]
        M[1, 2] += (nH / 2) - center[1]
        rotated = cv2.warpAffine(image, M, (nW, nH))
        return rotated



# 图像清晰度值
def getImageVar(image):
    img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageVar = cv2.Laplacian(img2gray, cv2.CV_64F).var()
    return imageVar

def sigmoid(x):
    return 1. / (1. + np.exp(-x) )

# 阶跃函数
def step_function(x):
    return np.array(x > 0, dtype=np.int)

# 高斯函数
def gaosi(sigma,x):
    return np.exp(-x**2/(2*sigma**2) )/(np.sqrt(2*np.pi)*sigma)

# 一维向量卷积
def conv_func(a, b, conv=True):
    if a.shape > b.shape:
        return conv_func( b, a)
    
    # Convert to np.array type
    a, b = list(map(np.array, [a, b]))
    
    # 反转
    if conv: 
        a = a[::-1] 
    res = []
    min_len, max_len = len(a), len(b)

    output_length = max_len + min_len - 1
    tmp = np.hstack((np.zeros(min_len-1), b, np.zeros(min_len-1)))

    # For each point, get the total sum of element-wise multiplication
    for i in range(output_length):
        val = np.sum(a * tmp[i:min_len+i])
        res.append(val)
    return np.array(res, dtype=a.dtype)


# 2维滤波器(卷积)
def my_filter2D( img , kel ):
    m,n = kel.shape
    # out 是输出
    out = np.zeros(img.shape)
    # mat 是补完 0 的
    mat = np.zeros( (img.shape[0]+2*m-2, img.shape[1]+2*n-2) )
    # 扩充
    mat[m-1:-(m-1),n-1:-(m-1)] = img
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i,j] = np.multiply(mat[i:i+m,j:j+n] , kel ).sum()    
    return out

# 测试 2 维卷积
""""
my_filter2D(np.ones((3,3)),np.ones((2,2)) )
"""

# 3维滤波器(卷积)
def my_filter3D( img , kel ):
    m,n = kel.shape
    # out 是输出
    out = np.zeros(img.shape)
    # mat 是补完 0 的
    mat = np.zeros( (img.shape[0]+2*m-2, img.shape[1]+2*n-2 , img.shape[2]) )
    # 扩充
    mat[m-1:-(m-1),n-1:-(m-1),:] = img
    
    # 对层数进行循环先
    for z in range(img.shape[2]):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                out[i,j,z] = np.multiply(mat[i:i+m,j:j+n,z] , kel ).sum()    
    return out

# 测试 3 维卷积
""""
my_filter3D(np.ones((2,3,4)),np.ones((2,2)) )[...,1]
"""

# 灰度图
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# 池化
def pooling(image , poolSize = 2 , poolStride = 2 , mode = 'max' ):  
    # image sizes
    in_row,in_col = np.shape(image)
    
    # outputMap sizes
    out_row,out_col = int(np.ceil( (in_row-poolSize)/poolStride +1 )),\
                      int(np.ceil( (in_col- poolSize)/poolStride +1 ))
    outputMap = np.zeros((out_row,out_col))
    # print(out_row,out_col)
    
    # 补 0
    row_remainder,col_remainder = np.mod(in_row ,poolStride),np.mod(in_col ,poolStride)
    # padding
    temp_map = np.pad(image, ((0,row_remainder),(0,col_remainder)),'edge')
    
    # pooling
    for r_idx in range(out_row):
        for c_idx in range(out_col):
            startY = r_idx * poolStride
            startX = c_idx * poolStride
            poolField = temp_map[startY:startY + poolSize, startX:startX + poolSize]
            if mode == 'max':
                poolOut = np.max(poolField)
            elif mode == 'mean':
                poolOut = np.mean(poolField)
                
            outputMap[r_idx,c_idx] = poolOut
    return  outputMap

"""
# 测试实例
test = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
print(test)
test_result = pooling(test, 2, 3, 'max')
print(test_result)
"""

# 2维　3维图像插值改变大小
def my_resize(src, size):
    dst_width, dst_height = size
    
    if len(src.shape) == 3:
        height, width, channels = src.shape
        if ((dst_height == height) and (dst_width == width)):
            return src
        dst_image = np.zeros((dst_height, dst_width, channels) )

        # Scale for resize.
        scale_x = float(width) /dst_width
        scale_y = float(height)/dst_height
        # tmp
        for k in range(channels):
            for dst_y in range(dst_height):
                for dst_x in range(dst_width):
                    # Original coords.
                    src_x = (dst_x + 0.5) * scale_x - 0.5
                    src_y = (dst_y + 0.5) * scale_y - 0.5
                    # INTER_LINEAR: 
                    # 2*2 neighbors.(对角线两个值)
                    src_x_0 = int(np.floor(src_x))
                    src_y_0 = int(np.floor(src_y))
                    src_x_1 = min(src_x_0 + 1, width - 1)
                    src_y_1 = min(src_y_0 + 1, height - 1)
                    # 插值(左上，右上，右下，左下，顺时针)
                    dst_image[dst_y, dst_x,k] = \
                    (src_y_1 - src_y) *  (src_x_1 - src_x) * src[src_y_0, src_x_0,k]\
                    + (src_y_1 - src_y) * (src_x - src_x_0) * src[src_y_0, src_x_1,k]\
                    + (src_y - src_y_0) * (src_x - src_x_0) * src[src_y_1, src_x_1,k]\
                    + (src_y - src_y_0) * (src_x_1 - src_x) * src[src_y_1, src_x_0,k]
        return dst_image
    
    elif len(src.shape) == 2:
        height, width = src.shape
        if ((dst_height == height) and (dst_width == width)):
            return src
        dst_image = np.zeros((dst_height, dst_width) )
        
        # Scale for resize.
        scale_x = float(width) /dst_width
        scale_y = float(height)/dst_height
        # tmp
        for dst_y in range(dst_height):
            for dst_x in range(dst_width):
                # Original coords.
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5
                # INTER_LINEAR: 
                # 2*2 neighbors.(对角线两个值)
                src_x_0 = int(np.floor(src_x))
                src_y_0 = int(np.floor(src_y))
                src_x_1 = min(src_x_0 + 1, width - 1)
                src_y_1 = min(src_y_0 + 1, height - 1)
                # 插值(左上，右上，右下，左下，顺时针)
                dst_image[dst_y, dst_x] = \
                (src_y_1 - src_y) *  (src_x_1 - src_x) * src[src_y_0, src_x_0]\
                + (src_y_1 - src_y) * (src_x - src_x_0) * src[src_y_0, src_x_1]\
                + (src_y - src_y_0) * (src_x - src_x_0) * src[src_y_1, src_x_1]\
                + (src_y - src_y_0) * (src_x_1 - src_x) * src[src_y_1, src_x_0]
        return dst_image

# 测试 resize 
my_resize(np.arange(16).reshape(4,4),(2,2) )


