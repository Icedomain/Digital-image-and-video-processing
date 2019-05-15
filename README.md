# Digital-image-and-video-processing
东南大学2019春季信号班学位课 
数字图像与视频处理 王桥
Copyright @ 180776 胡欣毅.  All rights reserved. 

### 学习指导及复习要点
- 由于修改不及时，所有版本代码、报告有不同处，按Python版本代码为准

# 课堂内容
## [1周]work1 初识opencv
1. 熟悉图像处理软件、平台，实现图像的读取与显示（例）
2. 图像像素级读写：求横坐标为25的所有像素三个通道像素值之和；（10/100）
3. 利用公式将RGB图像转化为灰度图，并显示；（20/100）
4. 对题目3得到的灰度图像进行二值化，给定上界A与下界B，在[A, B]内的像素值设置为255，反之为0；（20/100）
5. 图像顺时针旋转30度，并显示；（30/100）
6. 在图像中心截取像素为256*256的子图，并显示；（20/100）

参考[Python代码](/work1/Python/Task1.ipynb)

参考[C++代码](/work1/c++/hxy.cpp)

参考[报告](/work1/Task1.pdf)




## [2周]work2 边缘提取与特征匹配
0. 熟悉图像处理软件、平台，实现图像的读取与显示（例）
1. 【个人作业】课堂上的边缘提取算法的实现、并可视化（20/100）
2. 【小组作业】自行查找文献了解SIFT、SURF算法，并介绍（20/100）；
3. 【小组作业】实现SIFT算法（30/100）；
4. 【小组作业】实现SURF算法（附加题20）；
5. 【小组作业】测试SIFT算法，并可视化（10/100）；
6. 【小组作业】准备10页左右ppt，包括上述内容，并准备课堂抽查报告（20/100）。


注意(sift和surf的函数对opencv有一定要求，本人使用3.4.0.12版本):
```bash
$ conda activate hxy
$ pip install opencv-python==3.4.0.12 
$ pip install opencv-contrib-python==3.4.0.12
```
* 个人作业

参考[Python代码](/work2/Python/Task2.ipynb)

参考[C++代码](/work2/c++/hxy.cpp)

参考[报告](/work2/Task2.pdf)


* 小组作业一

参考[论文SIFT](/work2/SIFT.pdf)

参考[论文SURF](/work2/SURF.pdf)

参考[Python代码](/work2/team_work/sift.py)

参考[C++代码](/work2/team_work/c++/team1.cpp)

参考[PPT报告](/work2/team_work/ppt_sift_surf/zh.pdf)



## [3、4周]work3 图像清晰度恢复
1. 如何利用图像的边缘或SIFT或SURF特征，让模糊的图像变清晰。

提示:图像的边缘增强

参考[Python代码](/work3/Python/Task3.ipynb)

参考[C++代码](/work3/c++/hxy.cpp)

参考[报告](/work3/Task3.pdf)



## [5周]上课 一维数据边缘增强
上课随堂任务
1. sigmoid函数的边缘增强
2. 阶跃函数柔化后的边缘增强


参考[Python代码](/5周上课/class_task.ipynb)

参考[C++代码](/5周上课/c++/hxy.cpp)

参考[报告](/5周上课/class_task.pdf)


## [6周]上课 基于边缘检测的图像清晰度加强
上课随堂任务


参考[Python代码](/6周上课/class_task.ipynb)

参考[C++代码](/6周上课/c++/hxy.cpp)

参考[报告](/6周上课/class_task.pdf)


## [7周]上课 图像滤波+图像边缘检测+图像恢复(不调用opencv,函数自编)
上课随堂任务

1. 拍一张照片（具有清晰易检测边缘），压缩为255 * 255 大小。对照片进行2 * 2 max_pooling操作，滑动间隔为1 （30/100）

2. 对该照片进行laplace边缘检测（40/100）

3. 对该照片进行laplace图像增强（30/100）

注：
1. 本次实验不可调opencv库

2. 本次实验根据完成程度和完成速度评分，分数可能计入总评

参考[Python代码](/7周上课/class_task.ipynb)

参考[C++代码](/7周上课/c++/hxy.cpp)

参考[报告](/7周上课/class_task.pdf)



## [8周]work4 教材图像恢复算法实践

1. 利用Perona-Malik算法改善图像清晰度
2. 利用Gabor算法改善图像清晰度

参考[Python代码](/work4/Task4.ipynb)

参考[C++代码](/work4/c++/hxy.cpp)

参考[报告](/work4/Task4.pdf)

## [9周]上课随堂

1. 生成函数及边缘增强[同第5周任务]
2. 阶跃函数柔化和边缘增强[同第5周任务]
3. 卷积信号估计:从$ g(x)=f(x)*h(x) $ 中估计出　$ f(x)$
4. 图像矫正(仿射变换、透视变换)
5. 图像边缘增强
6. 课后补充
* 课后补充:自编图像矫正函数,提升速度

参考[Python代码](/9周上课/Python/class_task.ipynb)

参考[C++代码](/9周上课/c++/hxy.cpp)

参考[C++代码+霍夫变换](/9周上课/c++/wrap_hough.cpp)

参考[报告](/9周上课/class_task.pdf)

## [10周]五一放假
Pass


## [11周] 图像傅里叶变换及幅度、相角对图像的影响
1. 选两张图片,resize到同一尺寸
2. 图片进行傅里叶变换得到幅度和相角
3. 利用幅度、相角进行恢复、交叉恢复,判断幅度(模)与相位的重要性

结论：相位重要

参考[Python代码](/11周上课/Python/class_task.ipynb)

参考[C++代码](/11周上课/c++/hxy.cpp)

参考[报告](/11周上课/class_task.pdf)



## [12周] 







# 补充部分

* 各个程序中的函数集锦
basic 参考[Python代码](/my_definition/basic.py)
work4 Perona-Malik、Gabor算法　参考[Python代码](/my_definition/PM_gabor.py)
加噪去噪 参考[Python代码](/my_definition/noise_denoise.py)


* 

