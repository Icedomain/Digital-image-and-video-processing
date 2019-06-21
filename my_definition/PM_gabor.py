"""
220180776 胡欣毅
王桥　数字图像处理　自编函数库

"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

                    
def g(s,k):
    return np.exp(-(s/k)**2)

def get_c(grad_u,k):
    return g(abs(grad_u),k)

def PM1(src ,k = 12,lambd = .01 ):
    
    Ker_x = np.array([[-1, 0, 1], 
                      [-1, 0, 1], 
                      [-1, 0, 1]])
    Ker_y = np.array([[1, 1, 1], 
                      [0, 0, 0], 
                      [-1,-1,-1]])
    # Laplace扩展算子
    laplace = np.array([[1, 1, 1],
                        [1,-8, 1],
                        [1, 1, 1]])
    # grad_u
    grad_x = cv2.filter2D(src,  -1 , Ker_x )
    grad_y = cv2.filter2D(src,  -1 , Ker_y )
    grad_u = np.abs(grad_x)  + np.abs(grad_y) 
    # c
    c = get_c(grad_u,k = k)
    # grad_c
    grad_x_c = cv2.filter2D(src,  -1 , Ker_x )
    grad_y_c = cv2.filter2D(src,  -1 , Ker_y )
    grad_c = np.abs(grad_x_c)  + np.abs(grad_y_c)  
    # Laplace_u
    Laplace_u = cv2.filter2D(src,  -1 , laplace )
    # 
    im_out = (src + lambd*(grad_c * grad_u + c* Laplace_u ) )
    return im_out


def PM2(src ,k = 12,lambd = .001 ):
    
    edge = - np.array([[1, 1, 1],
                       [1,-8, 1],
                       [1, 1, 1]])
    # grad_u
    grad_u = cv2.filter2D(src,  -1 , edge )
    # c
    c = get_c(grad_u,k = k)
    # grad_c
    grad_c = cv2.filter2D(src,  -1 , edge )

    # Laplace_u
    x = cv2.getGaussianKernel(3 , 1.0)#高斯滤波器
    ker = x * x.T

    Laplace_u = cv2.filter2D(src,  -1 , ker )
    # out 
    im_out = (src + lambd*(grad_c * grad_u + c* Laplace_u ) )
    return im_out

def f(s,k):
    return 1.0 / (1.0 + (s / k) ** 2)


def Perona_Malik(src , times =30,dt=.01 ,kappa =12, option = 1 ):  
    ny,nx = src.shape
    src = src.astype('float')
    # copy
    res = src
    tmp = src
    
    # 迭代次数
    for t in range(times):
        # 一次迭代
        for i in range(ny):
            for j in range(nx):
                # 位置信息 边界处理
                iUp   =  max(0,i - 1)
                iDown =  min(ny-1,i + 1)
                jLeft =  max(0,j - 1)
                jRight = min(nx-1,j + 1)
                
                # 书本page216　先计算　deta_u
                deltaN = tmp[iUp,j] - tmp[i,j]
                deltaS = tmp[iDown,j] - tmp[i,j]
                deltaE = tmp[i,jRight] - tmp[i,j]
                deltaW = tmp[i,jLeft] - tmp[i,j]
                delta_u = np.array([deltaN , deltaS , deltaE , deltaW ])
                #print(delta_u)
                
                # 计算　c
                if (option == 1):
                    c = g(np.abs(delta_u),kappa)
                elif (option == 2):
                    c = f(np.abs(delta_u),kappa)
                # 相乘相加 加权赋值
                res[i,j] += dt * (sum(c * delta_u) )
        tmp = res            
    return res


def gabor_fn(sigma, theta, Lambda, psi, gamma):
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # ------这部分内容是为了确定卷积核的大小------
    # Bounding box
    nstds = 3 # Number of standard deviation sigma
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), \
               abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), \
               abs(nstds * sigma_y * np.cos(theta)))
    
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))
    # ------这部分内容是为了确定卷积核的大小------
    
    # Rotation 
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    # ------这部分正是上面的公式------
    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) \
                                        * np.cos(2 * np.pi / Lambda * x_theta + psi)
    return gb


#构建Gabor滤波器
# cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype) 参数说明
def build_filter():
    ksize = (5,5)
    lamda = np.pi/2.0 # 波长
    direction = 8
    out = np.zeros(ksize)
 
    for i,theta in enumerate(np.arange(0, np.pi, np.pi / direction)):
        # gabor方向，0度，22.5度, 45度，67.5度, 90度, 
        # 112.5度，135度，157.5度 共8个
        kern = cv2.getGaborKernel(ksize,.7,theta,lamda,\
                                  gamma = 0.5, psi= 0, ktype=cv2.CV_32F)
        #np.maximum(out,kern,out)
        out += kern
    out /= out.sum()
    return out
