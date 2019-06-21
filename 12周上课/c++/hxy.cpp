/* 
 220180776 胡欣毅
 数字图像处理　王桥
 课程实验十

 */

#include <iostream>
#include<string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void shift(Mat & magI)
{
    //如果有奇数行或列，则对频谱进行裁剪
    magI = magI(Rect(0, 0, magI.cols&-2, magI.rows&-2));

    //重新排列傅里叶图像中的象限，使得原点位于图像中心
    int cx = magI.cols / 2;
    int cy = magI.rows / 2;

    Mat q0(magI, Rect(0, 0, cx, cy));       //左上角图像划定ROI区域
    Mat q1(magI, Rect(cx, 0, cx, cy));      //右上角图像
    Mat q2(magI, Rect(0, cy, cx, cy));      //左下角图像
    Mat q3(magI, Rect(cx, cy, cx, cy));     //右下角图像

    //变换左上角和右下角象限
    Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    //变换右上角和左下角象限
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

}

Mat fft(Mat src)
{
    Mat planes[] = { src , Mat::zeros(src.size(),CV_64F) }; 
    Mat complexI; 
    merge(planes, 2, complexI); 
    dft(complexI, complexI); 
    split(complexI, planes);

    // 求幅度    
    Mat fft_abs  ;    
    magnitude(planes[0], planes[1], fft_abs);

    shift(fft_abs);
    //normalize(fft_abs,fft_abs,0,1,CV_MINMAX);
    return fft_abs ; 
}

void show_fft(string str,Mat fft_abs)
{   
    Mat img ; 
    fft_abs.copyTo(img) ; 
    img = Mat_<float>(img);
    img += Scalar::all(1);//转换到对数尺度
	log(img, img);//求自然对数

    imshow(str,img);
    img.convertTo(img, CV_8UC1, 255, 0);
    imwrite("../log.png",img);
}

int main() //int, char** argv )  
{   
/*
    cv::Mat src = imread( argv[1], cv::IMREAD_UNCHANGED);
    if( !src.data )
    { return -1; }
*/
    Mat im = cv::imread("../../lena_std.tif", -1);

    Mat gray ,gray_double;
    cvtColor(im, gray, CV_RGB2GRAY );
    gray_double = Mat_<double>(gray);

    // ########## 定义核 ########### //
    int  ker_size = 15 ;
    double sigma = 2.0 ;
    // 高斯滤波器
    Mat x = getGaussianKernel( ker_size , sigma  ) ;
    Mat ker = x * x.t() ;

    // def 位置
    int i = 500 ; 
    int j = 100 ;
    // int i = 250 , j = 250 ;

    //  取ROI并与图位置相乘
    int m = ker.rows ;
    int n = ker.cols ;
    
    Mat tmp = Mat::zeros(gray.size(), gray_double.type());
    cv::Rect ROI = Rect(i-int(m/2), j-int(n/2) , m,n);
    tmp(ROI) = gray_double(ROI).mul(ker) ; 

    circle(im, Point2f(i,j), 5, Scalar(0, 255, 0), 4);
    imshow("src",im);  

    //fft  of tmp
    Mat fft_abs = fft(tmp) ;
    imshow("abs",fft_abs);

    show_fft("log abs",fft_abs) ; 

    fft_abs.convertTo(fft_abs, CV_8UC1, 255, 0);
    imwrite("../fft.png",fft_abs);

    waitKey();  
    return 0;  
} 

