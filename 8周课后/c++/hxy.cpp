/* 
 220180776 胡欣毅
 数字图像处理　王桥
 课程实验七

 */

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

//给原图像增加椒盐噪声
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


double g(double s , float k)
{
  return exp(- pow((abs(s)/k),2) );
} 

double f(double s , float k)
{
  return 1.0 / (1.0 + pow((abs(s)/k),2) );
}


void Perona_Malik(Mat& src, Mat& dst, int iter, double dt, double kappa, int option)
{
	// 行列大小
	int nx = src.cols ; 
	int ny = src.rows ;
	// copy 一份
	Mat I_t ;
	Mat I_tmp ;
	src.convertTo(I_t , CV_64FC1);
	src.convertTo(I_tmp , CV_64FC1);

	for (int t = 0; t < iter; t++)
	{
		for (int i = 0; i < ny; i++)
		{
			for (int j = 0; j < nx; j++)
			{
			// 加边界处理
			int iUp = max(0,i - 1) ; 
			int iDown = min(ny - 1,i + 1);
			int jLeft = max(0,j - 1 );
			int jRight = min(nx - 1,j + 1);    
	
			double deltaN = I_t.at<double>(iUp,j) - I_t.at<double>(i,j);
			double deltaS = I_t.at<double>(iDown,j) - I_t.at<double>(i,j);
			double deltaE = I_t.at<double>(i,jRight) - I_t.at<double>(i,j);
			double deltaW = I_t.at<double>(i,jLeft) - I_t.at<double>(i,j);
	
			double cN, cS, cE, cW;
			if (1 == option){
				cN = g(deltaN , kappa);
				cS = g(deltaS , kappa);
				cE = g(deltaE , kappa);
				cW = g(deltaW , kappa);
				}
			else if (2 == option){
				cN = f(deltaN , kappa);
				cS = f(deltaS , kappa);
				cE = f(deltaE , kappa);
				cW = f(deltaW , kappa);
				}
	
			I_tmp.at<double>(i,j) += dt * (cN * deltaN + cS * deltaS + cE * deltaE + cW * deltaW);
			}
		}  
	// 一次迭代完成
	I_t = I_tmp ;
	}
	I_tmp.convertTo(dst, CV_8UC1);
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

    Mat gray;
    cvtColor( src, gray, CV_RGB2GRAY );
    imshow("gray Image", gray);
	imwrite("../gray.jpg", gray);


    double mu = 2 ;
    double sigma = 0.8 ;
  	Mat gray_add = addGaussianNoise(gray, mu,sigma);
 	imshow("gray Image add noise", gray_add);
    
    float k = 0.003;
    double dt = 14;
/*     cout<<"输入步长dt"<<"输入ｋ"<<endl;
    cin>>dt>>k; */
    
	// 0.0008
    // 15

    Mat pm ;
	// Perona_Malik(gray_add, pm, 40, dt, k,1) ;
	Perona_Malik(gray, pm, 40, dt, k,1) ;
    imshow("Perona Malik算法", pm);  
	imwrite("../pm.jpg", pm);

    waitKey();  
    return 0;  
} 

