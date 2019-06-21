/* 
 220180776 胡欣毅
 数字图像处理　王桥
 课程实验五

 */

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
 
int main() //int, char** argv )  
{   
/*
    cv::Mat src = imread( argv[1], cv::IMREAD_UNCHANGED);
    if( !src.data )
    { return -1; }
*/
    cv::Mat src = cv::imread("../../data/4.1.05.tiff", cv::IMREAD_UNCHANGED);	 
    //cv::imshow("src Image", src);

    Mat gray;
    cvtColor( src, gray, CV_RGB2GRAY );
    //cv::imshow("gray Image", gray);
    imwrite("../gray.jpg",gray);

    // 低通滤波器 模糊化
    Mat low_filter = (Mat_<double>(3,3) << 1,1,1,
                                         1,1,1,
                                         1,1,1)/9.0 ;
    Mat f_out;
    filter2D(gray, f_out, -1 , low_filter);


    // 边缘检测
    cv::Mat  Laplace = (Mat_<double>(3,3) << -1,-1,-1,
                                              -1,8,-1,
                                              -1,-1,-1);
    Mat deta_f;
    filter2D(f_out, deta_f, -1 , Laplace);  


    double lambd = .5;
    Mat re_build = f_out + lambd * deta_f ;
    imwrite("../re_build.jpg",re_build);


    // 高斯滤波器 模糊化
    Mat gaosi_filter = (Mat_<double>(5,5) <<1,4,7,4,1,
                                            4,16,26,16,4,
                                            7,26,41,26,7,
                                            4,16,26,16,4,
                                            1,4,7,4,1) / 273.0 ;

    filter2D(gray, f_out, -1 , gaosi_filter);

    // 边缘检测

    filter2D(f_out, deta_f, -1 , Laplace);  


    lambd = .5;
    re_build = f_out + lambd * deta_f ;
    imwrite("../re_build_gass.jpg",re_build);


    waitKey();  
    return 0;  
} 

