/* 
 220180776 胡欣毅
 数字图像处理　王桥
 课程实验六

 */

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat pooling(Mat& img ,int poolSize = 2 , int poolStride = 2 , string mode = "max" )
{
    // output sizes
    Mat pool_img = Mat::zeros((int)((img.rows - poolSize) / poolStride + 1), 
                              (int)((img.cols - poolSize) / poolStride + 1), CV_8UC1);
    int pool_row = pool_img.rows;
    int pool_col = pool_img.cols;

    // pad 
    Mat tmp = Mat::zeros(img.rows + poolSize ,img.cols + poolSize , CV_8UC1);
    for(int i = 0 ;i<img.rows;i++){
        for(int j = 0;j<img.cols;j++){
            tmp.at<uchar>(i,j) = img.at<uchar>(i,j) ;
        }
    }

    // pooling
    if(mode == "max"){
        for(int r_idx = 0 ; r_idx<pool_row;r_idx++){
            for(int c_idx = 0;c_idx<pool_col;c_idx++){
                int startY = r_idx * poolStride;
                int startX = c_idx * poolStride;  
                // int srcHeight, srcWidth, subHeight, subWidth;
                Mat poolField = tmp(Rect(startY,startX,poolSize,poolSize) );
                
                int max = 0;
                for (int i = 0;i<poolSize;i++){
                    for (int j = 0;j<poolSize;j++){
                        if (poolField.at<uchar>(i,j)>max ) max = poolField.at<uchar>(i,j);
                    }
                }       
                pool_img.at<uchar>(c_idx,r_idx) = max ;
            }
        }
    }
    else if(mode == "mean")
    {
        for(int r_idx = 0 ; r_idx<pool_row;r_idx++){
            for(int c_idx = 0;c_idx<pool_col;c_idx++){
                int startY = r_idx * poolStride;
                int startX = c_idx * poolStride;  
                // int srcHeight, srcWidth, subHeight, subWidth;
                Mat poolField = tmp(Rect(startY,startX,poolSize,poolSize) ) ;

                int sum = 0;
                for (int i = 0;i<poolSize;i++){
                    for (int j = 0;j<poolSize;j++){
                        sum += poolField.at<uchar>(i,j) ;
                    }
                }
                pool_img.at<uchar>(c_idx,r_idx) = sum / (poolSize*poolSize) ;
            }
        }
    }
    return pool_img ;
}

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
    imwrite("../gray.jpg",gray);

    int width = 255;
    int height = 255;
    Mat im_resize;
    resize(gray, im_resize, cv::Size(width, height));
    imwrite("../resize.jpg",im_resize);
    // cv::imshow("resize Image", im_resize);

    // 池化
    Mat im_pool = pooling(im_resize, 2 , 1, "max");
    //imshow("pooling Img",im_pool);
    imwrite("../pooling.jpg",im_pool);


    // 低通滤波器 模糊化
    Mat low_filter = (Mat_<double>(3,3) << 1,1,1,
                                           1,1,1,
                                           1,1,1)/9.0 ;
    Mat f_out;
    filter2D(im_pool, f_out, -1 , low_filter);


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

    filter2D(im_pool, f_out, -1 , gaosi_filter);

    // 边缘检测

    filter2D(f_out, deta_f, -1 , Laplace);  


    lambd = .5;
    re_build = f_out + lambd * deta_f ;
    imwrite("../re_build_gass.jpg",re_build);


    waitKey();  
    return 0;  
} 

