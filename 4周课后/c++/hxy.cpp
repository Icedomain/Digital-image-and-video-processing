/* 
 220180776 胡欣毅
 数字图像处理　王桥
 课程实验三

 */

#include <iostream>
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
    cv::Mat src = cv::imread("../../hudie/3.bmp", -1);	
    //cv::imshow("src Image", src);

    Mat gray;
    cvtColor( src, gray, CV_RGB2GRAY );
    //cv::imshow("gray Image", gray);
    imwrite("../gray.jpg",gray);

    // 边缘检测
    cv::Mat  kernel_1 = (Mat_<double>(3,3) << -1,-1,-1,
                                              -1,8,-1,
                                              -1,-1,-1);
    Mat edges;
    filter2D(gray, edges, -1 , kernel_1);  
    imwrite("../edges.jpg",edges);

    double lambd = .4;
    Mat strengthen = gray + lambd * edges ;
    imwrite("../strengthen.jpg",strengthen);


    // 低通滤波器
    Mat kernel_2 = (Mat_<double>(3,3) << 1,1,1,
                                         1,1,1,
                                         1,1,1)/9.0 ;
    Mat rect;
    filter2D(src, rect, -1 , kernel_2);
    // imwrite("../rect.jpg",rect);


    // 高斯滤波器
    Mat kernel_3 = (Mat_<double>(5,5) <<1,4,7,4,1,
                                        4,16,26,16,4,
                                        7,26,41,26,7,
                                        4,16,26,16,4,
                                        1,4,7,4,1)/273.0 ;
    Mat gaussian;
    filter2D(src, gaussian, -1 , kernel_3);
    // imwrite("../gaussian.jpg",gaussian);


    // 锐化滤波器
    Mat kernel_sharpen_1 = (Mat_<double>(3,3) << -1,-1,-1,
                                              -1,9,-1,
                                              -1,-1,-1);
    Mat kernel_sharpen_2 = (Mat_<double>(3,3) << 0,-2,0,
                                            -2,9,-2,
                                            0,-2,0);
    Mat kernel_sharpen_3 = (Mat_<double>(5,5) << -1,-1,-1,-1,-1,
                                            -1,2,2,2,-1,
                                            -1,2,8,2,-1,
                                            -1,2,2,2,-1, 
                                            -1,-1,-1,-1,-1)/8.0;
    Mat output_1,output_2,output_3;
    filter2D(src, output_1, -1 , kernel_sharpen_1);
    filter2D(src, output_2, -1 , kernel_sharpen_2);
    filter2D(src, output_3, -1 , kernel_sharpen_3);

    // 显示锐化效果
    imwrite("../sharpen_1.png",output_1);
    imwrite("../sharpen_2.png",output_2);
    imwrite("../sharpen_3.png",output_3);

    waitKey();  
    return 0;  
} 

