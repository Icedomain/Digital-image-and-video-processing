/* 
 220180776 胡欣毅
 数字图像处理　王桥
 课程实验二

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

    cv::Mat src = cv::imread("../../data//4.1.05.tiff", cv::IMREAD_UNCHANGED);	 
    cv::imshow("src", src);

    Mat gray;
    cvtColor( src, gray, CV_RGB2GRAY );
    cv::imshow("gray", gray);
    


    Mat grad_x, grad_y;
  	Mat abs_grad_x, abs_grad_y, grad;

    // 一阶方法
    // Roberts
    cv::Mat Ker_x,Ker_y;
    Ker_x = (Mat_<double>(2,2) << 1,0,
                                  0,-1);
    Ker_y = (Mat_<double>(2,2) << 0,-1,
                                  1,0);    

    filter2D(gray, grad_x, -1 , Ker_x);  
    filter2D(gray, grad_y, -1 , Ker_y);
    convertScaleAbs( grad_x, abs_grad_x );
    convertScaleAbs( grad_y, abs_grad_y );  
    addWeighted( abs_grad_x, 1.0, abs_grad_y, 1.0, 0, grad ); 
    grad = (grad > 25)*255; 
    imshow("Roberts", grad);


    Mat Kernelx,Kernely;

    // Prewwitt   
    Kernelx = (Mat_<double>(3,3) << -1, 0, 1,
                                    -1, 0, 1, 
                                    -1, 0, 1);
    Kernely = (Mat_<double>(3,3) << 1, 1, 1,
                                    0, 0, 0,
                                    -1, -1, -1);  

    filter2D(gray, grad_x, -1 , Kernelx);  
    filter2D(gray, grad_y, -1 , Kernely);
    convertScaleAbs( grad_x, abs_grad_x );
    convertScaleAbs( grad_y, abs_grad_y );  
    addWeighted( abs_grad_x, 1.0, abs_grad_y, 1.0, 0, grad ); 
    grad = (grad > 60)*255; 
    imshow("Prewwitt", grad);  

    // Sobel 核化滤波器
    Kernelx = (Mat_<double>(3,3) << -1, 0, 1,
                                    -2, 0, 2,
                                    -1, 0, 1); 
    Kernely = (Mat_<double>(3,3) << 1, 2, 1, 
                                    0, 0, 0, 
                                    -1, -2, -1);

    filter2D(gray, grad_x, -1 , Kernelx);  
    filter2D(gray, grad_y, -1 , Kernely);
    convertScaleAbs( grad_x, abs_grad_x );
    convertScaleAbs( grad_y, abs_grad_y );   
    addWeighted( abs_grad_x, 1.0, abs_grad_y, 1.0, 0, grad ); 
    grad = (grad > 100)*255;
    imshow("Sobel", grad);  

    // Sobel　切用函数版   
    // Gradient X
    Sobel( gray, grad_x, -1, 1, 0,  3);
    convertScaleAbs( grad_x, abs_grad_x );
    // Gradient Y
    Sobel( gray, grad_y, -1, 0, 1,  3);
    convertScaleAbs( grad_y, abs_grad_y );
    // Total Gradient (approximate)
    addWeighted( abs_grad_x, 1.0, abs_grad_y, 1.0, 0, grad );
    grad = (grad > 100)*255;
    imshow("roberts use function", grad);  


    // 二阶方法
    // Laplace 横竖都一样
    Mat abs_grad;
    Mat Kernel = (Mat_<double>(3,3) << 0,1,0,
                                       1,-4,1,
                                       0,1,0 ); 

    filter2D(gray, grad, -1 , Kernel);  
    convertScaleAbs( grad, abs_grad );
    abs_grad = (abs_grad > 40)*255;
    imshow("Laplace", abs_grad);  


    // LOG 横竖都一样
    Mat Kernel_LOG = (Mat_<double>(5,5) << 0,0,1,0,0,
                                           0,1,2,1,0,
                                           1,2,-16,2,1,
                                           0,1,2,1,0,
                                           0,0,1,0,0  ); 

    filter2D(gray, grad, -1 , Kernel_LOG);  
    convertScaleAbs( grad, abs_grad );  
    abs_grad = (abs_grad > 165)*255;
    //cout <<abs_grad<<endl;

    imshow("LOG", abs_grad);  

    //  Canny
    Mat canny_filter;
    Canny(gray, canny_filter, 80, 180, 3);
    imshow("Canny", canny_filter);  

    waitKey();  
    return 0;  
    
} 

