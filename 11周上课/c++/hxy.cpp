/* 
 220180776 胡欣毅
 数字图像处理　王桥
 课程实验九

 */
// https://blog.csdn.net/lindamtd/article/details/80943747

#include<iostream>
#include<string>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

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

void show_fft(string str,Mat fft_abs)
{   
    Mat img ; 
    fft_abs.copyTo(img) ; 
    img += Scalar::all(1);//转换到对数尺度
	log(img, img);//求自然对数
    shift(img);

    normalize(img,img,0,1,CV_MINMAX);
    imshow(str,img);

}


int main() //int, char** argv )  
{   
/*
    cv::Mat src = imread( argv[1], -1);
    if( !src.data )
    { return -1; }
*/
    Mat im1 = cv::imread("../../data/4.1.05.tiff", -1);	  
    Mat im2 = cv::imread("../../data/4.2.07.tiff", -1);
    Mat im2_resize ;
    resize(im2, im2_resize, im1.size());
    
    Mat gray1 , gray2 ;
    cvtColor(im1, gray1, CV_RGB2GRAY );
    cvtColor(im2_resize, gray2, CV_RGB2GRAY );

    // ############################# //
    
    Mat planes1[] = { Mat_<float>(gray1), Mat::zeros(gray1.size(),CV_32F) }; 
    Mat planes2[] = { Mat_<float>(gray2), Mat::zeros(gray2.size(),CV_32F) }; 
    Mat complexI1 , complexI2; 
    //将planes融合合并成一个多通道数组complexI  一通道实部　一通道虚部
    merge(planes1, 2, complexI1); 
    merge(planes2, 2, complexI2); 

    // //进行傅里叶变换
    dft(complexI1, complexI1); 
    dft(complexI2, complexI2);
    
    cout<<complexI1.channels()<<'\t'<<complexI2.channels()<<endl;

    //planes[0] = Re(DFT(I),planes[1] = Im(DFT(I))
    //planes[0]为实部,planes[1]为虚部
    Mat im1_fft_abs , im1_fft_angle , im2_fft_abs , im2_fft_angle ; 
    split(complexI1, planes1);
    split(complexI2, planes2);
    // 求幅度        
    magnitude(planes1[0], planes1[1], im1_fft_abs);
    magnitude(planes2[0], planes2[1], im2_fft_abs);

    // 求相角
    phase(planes1[0], planes1[1], im1_fft_angle);
    phase(planes2[0], planes2[1], im2_fft_angle);

    show_fft("im1 fft abs" , im1_fft_abs) ;
    show_fft("im2 fft abs" , im2_fft_abs) ;

    // dst1 ~dst4
    // 恢复
    // dst1 :im1_fft_abs im1_fft_angle
    // dst2 :im2_fft_abs im2_fft_angle
    // 交叉
    // dst3 :im1_fft_abs im2_fft_angle
    // dst4 :im2_fft_abs im1_fft_angle
    
    Mat real = Mat::zeros(gray1.size(),CV_32F) ;
    Mat imag = Mat::zeros(gray1.size(),CV_32F) ;
    Mat dst1 , dst2 ,dst3 , dst4; 
    Mat iDft[] = { Mat::zeros(planes1[0].size(), CV_32F), Mat::zeros(planes1[0].size(), CV_32F) };
    Mat newmat;

    // dst1 
    polarToCart(im1_fft_abs , im1_fft_angle ,real , imag );
    Mat mat1[2] = { real , imag } ;
    merge(mat1 , 2, newmat);     
    idft(newmat,newmat,DFT_REAL_OUTPUT);  
    split(newmat, iDft);//结果貌似也是复数
	magnitude(iDft[0], iDft[1], dst1);

    normalize(dst1,dst1,0,1,CV_MINMAX);
    cout << dst1.channels()<<endl;


    // dst2 
    polarToCart(im2_fft_abs , im2_fft_angle ,real , imag );
    Mat mat2[2] = { real , imag } ;
    merge(mat2 , 2, newmat);     
    idft(newmat,newmat,DFT_REAL_OUTPUT);  
    split(newmat, iDft);//结果貌似也是复数
	magnitude(iDft[0], iDft[1], dst2);

    normalize(dst2,dst2,0,1,CV_MINMAX);
    cout << dst2.channels()<<endl;

    // dst3 
    polarToCart(im1_fft_abs , im2_fft_angle ,real , imag );
    Mat mat3[2] = { real , imag } ;
    merge(mat3 , 2, newmat);     
    idft(newmat,newmat,DFT_REAL_OUTPUT);  
    split(newmat, iDft);//结果貌似也是复数
	magnitude(iDft[0], iDft[1], dst3);

    normalize(dst3,dst3,0,1,CV_MINMAX);
    cout << dst3.channels()<<endl;

    // dst4
    polarToCart(im2_fft_abs , im1_fft_angle ,real , imag );
    Mat mat4[2] = { real , imag } ;
    merge(mat4 , 2, newmat);     
    idft(newmat,newmat,DFT_REAL_OUTPUT);  
    split(newmat, iDft);//结果貌似也是复数
	magnitude(iDft[0], iDft[1], dst4);

    normalize(dst4,dst4,0,1,CV_MINMAX);
    cout << dst4.channels()<<endl;

    imshow("dst1",dst1);  
    imshow("dst2",dst2);  
    imshow("dst3",dst3);  
    imshow("dst4",dst4);  

    dst1.convertTo(dst1, CV_8UC1, 255, 0);
    dst2.convertTo(dst2, CV_8UC1, 255, 0);
    dst3.convertTo(dst3, CV_8UC1, 255, 0);
    dst4.convertTo(dst4, CV_8UC1, 255, 0);
    imwrite("../dst1.jpg",dst1);
    imwrite("../dst2.jpg",dst2);
    imwrite("../dst3.jpg",dst3);
    imwrite("../dst4.jpg",dst4);

    waitKey();  
    return 0;  
} 

