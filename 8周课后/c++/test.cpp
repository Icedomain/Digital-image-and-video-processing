/* 
 220180776 胡欣毅
 数字图像处理　王桥
 课程实验补充部分

 */

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;


double g(double s , float k)
{
  return exp(- pow((abs(s)/k),2) );
} 

double f(double s , float k)
{
  return 1.0 / (1.0 + pow((abs(s)/k),2) );
}

//给图像添加椒盐噪声
Mat addSaltNoise(Mat src, int n)
{
	Mat result = src.clone();
	for (int k = 0; k < n; k++)
	{
		//随机选取行列值
		int i = rand() % result.cols;
		int j = rand() % result.rows;
		if (result.channels() == 1)
		{
			result.at<uchar>(j, i) = 255;
		}
		else
		{
			result.at<Vec3b>(j, i)[0] = 255;
			result.at<Vec3b>(j, i)[1] = 255;
			result.at<Vec3b>(j, i)[2] = 255;
		}
 
	}
	return result;
}

//给图像添加高斯噪声
double generateGaussianNoise(double mu, double sigma)
{
	//定义最小值
	double epsilon = numeric_limits<double>::min();
	double z0=0, z1=0;
	bool flag = false;
	flag = !flag;
	if (!flag)
		return z1*sigma + mu;
	double u1, u2;
	do
	{
		u1 = rand()*(1.0 / RAND_MAX);
		u2 = rand()*(1.0/RAND_MAX);
	} while (u1 <= epsilon);
	z0 = sqrt(-2.0*log(u1))*cos(2*CV_PI*u2);
	z1 = sqrt(-2.0*log(u1))*sin(2 * CV_PI*u2);
	return z0*sigma + mu;
}
 
Mat addGaussianNoise(Mat& src ,double mu, double sigma) 
{
	Mat result = src.clone();
	int channels = result.channels();
	int nRows = result.rows;
	int nCols = result.cols*channels;
	if (result.isContinuous())
	{
		nCols = nCols*nRows;
		nRows = 1;
	}
	for (int i = 0; i < nRows; i++)
	{
		for (int j = 0; j < nCols; j++)
		{
			int val = result.ptr<uchar>(i)[j] + generateGaussianNoise(mu, sigma)*32;
			if (val < 0)
				val = 0;
			if (val > 255)
				val = 255;
			result.ptr<uchar>(i)[j] = (uchar)val;
		}
	}
	return result;
}

void PM(Mat& img, float K, float t)
{
	Mat img_copy = img.clone();
	float gradu[4];
	float c[4];
	float sum = 0;
	for (int i = 1; i < img_copy.rows - 1; ++i)
	{
		for (int j = 1; j < img_copy.cols - 1; ++j)
		{
			sum = 0;
			gradu[0] = img_copy.ptr<float>(i - 1)[j] - img_copy.ptr<float>(i)[j];
			gradu[1] = img_copy.ptr<float>(i + 1)[j] - img_copy.ptr<float>(i)[j];
			gradu[2] = img_copy.ptr<float>(i)[j-1] - img_copy.ptr<float>(i)[j];
			gradu[3] = img_copy.ptr<float>(i)[j + 1] - img_copy.ptr<float>(i)[j];
			for (int k = 0; k < 4; ++k)
			{
				c[k] = g(gradu[k], K);
				sum += c[k] * gradu[k] / 2;
			}
			img.ptr<float>(i)[j] = t * sum + img_copy.ptr<float>(i)[j];
		}
	}

}



int main() //int, char** argv )  
{   
/*
    cv::Mat src = imread( argv[1], cv::IMREAD_UNCHANGED);
    if( !src.data )
    { return -1; }
*/

    Mat src = cv::imread("../../hudie/3.bmp", cv::IMREAD_UNCHANGED);	
    //cv::imshow("src Image", src);

    Mat gray ;	
    cvtColor( src, gray, CV_RGB2GRAY );
    imshow("gray Image", gray);
	imwrite("../gray.jpg", gray);

    double mu = 1.0 ;
    double sigma = 0.8 ;

    Mat gray_add_noise = addGaussianNoise(gray, mu,sigma);
    double psnr = PSNR(gray , Mat_<uchar>(gray_add_noise) );
    cout << "noise_psnr= " << psnr << endl;
    imshow("noise Image", gray_add_noise);
    imwrite("../noise.jpg", gray_add_noise);
 
	float t = 0.01;//一次项展开系数
	float K = 15;  //g函数参数
	int N = 40; //迭代次数

    gray_add_noise = Mat_<float>(gray_add_noise);
    Mat dst =  gray_add_noise ;

	for (int i = 0; i < N; i++)
	{
		PM(dst, K, t);
		psnr = PSNR(gray, Mat_<uchar>(dst));
		cout << "processed_psnr= " << psnr << endl;
	}

    dst = Mat_<uchar>(dst);
    imshow("PM Image", dst);
	imwrite("../pm.jpg", dst);

    waitKey();  
    return 0;  
} 




