
/* 
 220180776 胡欣毅
 数字图像处理　王桥
 课程实验八
 9周 homework 部分

 */

// 参考　https://blog.csdn.net/xiaowei_cqu/article/details/26471527
// 插值参考　https://cniter.github.io/posts/e124baa1.html#fn1

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


// 仿射变换
void my_warpAffine(cv::Mat& src, cv::Mat& dst, int x_len, int y_len , Mat M )
{
	// 行是y 列是x
	int dst_rows = y_len;
	int dst_cols = x_len;

	if (src.channels() == 1) {
		dst = cv::Mat::zeros(dst_rows, dst_cols, CV_8UC1); //灰度图初始
	}
	else {
		dst = cv::Mat::zeros(dst_rows, dst_cols, CV_8UC3); //RGB图初始
	}
 

	cv::Mat M_inv = M.inv(); // 求逆矩阵
 
	for (int i = 0; i < dst.rows; i++){ // i = y
		for (int j = 0; j < dst.cols; j++){ // j = x 
			cv::Mat dst_coordinate = (cv::Mat_<double>(3, 1) << j, i, 1);
			cv::Mat src_coordinate = M_inv * dst_coordinate ;

			double v = src_coordinate.at<double>(0, 0); // 原图像的横坐标，列，宽
			double w = src_coordinate.at<double>(1, 0); // 原图像的纵坐标，行，高
 
			//双线性插值
			// 判断是否越界

			if (v < 0) v = 0; 
			if (v > src.cols - 1) v = src.cols - 1;
			if (w < 0) w = 0; 
			if (w > src.rows - 1) w = src.rows - 1; 
 
			if (v >= 0 && w >= 0 && v <= src.cols - 1 && w <= src.rows - 1)
			{
				int top = floor(w), bottom = ceil(w), left = floor(v), right = ceil(v); //与映射到原图坐标相邻的四个像素点的坐标
				double pw = w - top; //pw为坐标 行 的小数部分(坐标偏差)
				double pv = v - left; //pv为坐标 列 的小数部分(坐标偏差)
				if (src.channels() == 1){
					//灰度图像
					dst.at<uchar>(i, j) = (1 - pw)*(1 - pv)*src.at<uchar>(top, left) + (1 - pw)*pv*src.at<uchar>(top, right) 
					+ pw*(1 - pv)*src.at<uchar>(bottom, left) + pw*pv*src.at<uchar>(bottom, right);
				}
				else{
					//彩色图像
					dst.at<cv::Vec3b>(i, j)[0] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[0] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[0] 
					+ pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[0] + pw*pv*src.at<cv::Vec3b>(bottom, right)[0];
					dst.at<cv::Vec3b>(i, j)[1] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[1] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[1] 
					+ pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[1] + pw*pv*src.at<cv::Vec3b>(bottom, right)[1];
					dst.at<cv::Vec3b>(i, j)[2] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[2] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[2] 
					+ pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[2] + pw*pv*src.at<cv::Vec3b>(bottom, right)[2];
				}
			}
		}
	}
}

// 透视变换
void my_warpPerspective(cv::Mat& src, cv::Mat& dst, int x_len, int y_len , Mat M )
{
	// 行是y 列是x
	int dst_rows = y_len;
	int dst_cols = x_len;

	if (src.channels() == 1) {
		dst = cv::Mat::zeros(dst_rows, dst_cols, CV_8UC1); //灰度图初始
	}
	else {
		dst = cv::Mat::zeros(dst_rows, dst_cols, CV_8UC3); //RGB图初始
	}
 
	for (int i = 0; i < dst.rows; i++){ // i = y
		for (int j = 0; j < dst.cols; j++){ // j = x 
			cv::Mat A = (cv::Mat_<double>(2, 2) << M.at<double>(0,0) - j * M.at<double>(2,0), 
												   M.at<double>(0,1) - j * M.at<double>(2,1),
													M.at<double>(1,0) - i * M.at<double>(2,0),
													M.at<double>(1,1) - i * M.at<double>(2,1)																										
													);
			cv::Mat b = (cv::Mat_<double>(2, 1) <<  M.at<double>(0,2) - j * M.at<double>(2,2),
													M.at<double>(1,2) - i * M.at<double>(2,2)
													 );
			cv::Mat src_coordinate;
			solve(A, -1*b, src_coordinate,CV_LU);
			// src_coordinate = -1*(A.inv())*b;

			double v = src_coordinate.at<double>(0, 0); // 原图像的横坐标，列，宽
			double w = src_coordinate.at<double>(1, 0); // 原图像的纵坐标，行，高

			//双线性插值
			// 判断是否越界

			if (v < 0) v = 0; 
			if (v > src.cols - 1) v = src.cols - 1;
			if (w < 0) w = 0; 
			if (w > src.rows - 1) w = src.rows - 1; 
 
			if (v >= 0 && w >= 0 && v <= src.cols - 1 && w <= src.rows - 1)
			{
				int top = floor(w), bottom = ceil(w), left = floor(v), right = ceil(v); //与映射到原图坐标相邻的四个像素点的坐标
				double pw = w - top; //pw为坐标 行 的小数部分(坐标偏差)
				double pv = v - left; //pv为坐标 列 的小数部分(坐标偏差)
				if (src.channels() == 1){
					//灰度图像
					dst.at<uchar>(i, j) = (1 - pw)*(1 - pv)*src.at<uchar>(top, left) + (1 - pw)*pv*src.at<uchar>(top, right) 
					+ pw*(1 - pv)*src.at<uchar>(bottom, left) + pw*pv*src.at<uchar>(bottom, right);
				}
				else{
					//彩色图像
					dst.at<cv::Vec3b>(i, j)[0] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[0] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[0] 
					+ pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[0] + pw*pv*src.at<cv::Vec3b>(bottom, right)[0];
					dst.at<cv::Vec3b>(i, j)[1] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[1] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[1] 
					+ pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[1] + pw*pv*src.at<cv::Vec3b>(bottom, right)[1];
					dst.at<cv::Vec3b>(i, j)[2] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[2] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[2] 
					+ pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[2] + pw*pv*src.at<cv::Vec3b>(bottom, right)[2];
				}
			}
		}
	}
}

// 透视变换 优化版
void my_warpPerspective_opt(cv::Mat& src, cv::Mat& dst, int x_len, int y_len , Mat M )
{
	// 行是y 列是x
	int dst_rows = y_len;
	int dst_cols = x_len;

	if (src.channels() == 1) {
		dst = cv::Mat::zeros(dst_rows, dst_cols, CV_8UC1); //灰度图初始
	}
	else {
		dst = cv::Mat::zeros(dst_rows, dst_cols, CV_8UC3); //RGB图初始
	}
 

	cv::Mat M_inv = M.inv(); // 求逆矩阵
 
	for (int i = 0; i < dst.rows; i++){ // i = y
		for (int j = 0; j < dst.cols; j++){ // j = x 
			cv::Mat dst_coordinate = (cv::Mat_<double>(3, 1) << j, i, 1);
			cv::Mat src_coordinate = M_inv * dst_coordinate ;

			double v = src_coordinate.at<double>(0, 0)/src_coordinate.at<double>(2, 0); // 原图像的横坐标，列，宽
			double w = src_coordinate.at<double>(1, 0)/src_coordinate.at<double>(2, 0); // 原图像的纵坐标，行，高
 
			//双线性插值
			// 判断是否越界

			if (v < 0) v = 0; 
			if (v > src.cols - 1) v = src.cols - 1;
			if (w < 0) w = 0; 
			if (w > src.rows - 1) w = src.rows - 1; 
 
			if (v >= 0 && w >= 0 && v <= src.cols - 1 && w <= src.rows - 1)
			{
				int top = floor(w), bottom = ceil(w), left = floor(v), right = ceil(v); //与映射到原图坐标相邻的四个像素点的坐标
				double pw = w - top; //pw为坐标 行 的小数部分(坐标偏差)
				double pv = v - left; //pv为坐标 列 的小数部分(坐标偏差)
				if (src.channels() == 1){
					//灰度图像
					dst.at<uchar>(i, j) = (1 - pw)*(1 - pv)*src.at<uchar>(top, left) + (1 - pw)*pv*src.at<uchar>(top, right) 
					+ pw*(1 - pv)*src.at<uchar>(bottom, left) + pw*pv*src.at<uchar>(bottom, right);
				}
				else{
					//彩色图像
					dst.at<cv::Vec3b>(i, j)[0] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[0] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[0] 
					+ pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[0] + pw*pv*src.at<cv::Vec3b>(bottom, right)[0];
					dst.at<cv::Vec3b>(i, j)[1] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[1] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[1] 
					+ pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[1] + pw*pv*src.at<cv::Vec3b>(bottom, right)[1];
					dst.at<cv::Vec3b>(i, j)[2] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[2] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[2] 
					+ pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[2] + pw*pv*src.at<cv::Vec3b>(bottom, right)[2];
				}
			}
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

    Mat src = cv::imread("../../desktop.jpeg", cv::IMREAD_UNCHANGED);	
    // cv::imshow("src Image", src);

    Mat gray;
    cvtColor( src, gray, CV_RGB2GRAY );
    // cv::imshow("gray Image", gray);


	/*###############　角点检测　###############*/
  
     int x_len = 1128 - 345;
	 int y_len = 780 - 189;   


	/*###############　仿射变换　###############*/
    Point2f srcpoints[3] ={ 
        cv::Point2f(345, 189),
        cv::Point2f(322, 763),
        cv::Point2f(1140, 780) 
        };
  
    Point2f dstpoints[3] = {
        cv::Point2f(0, 0),
        cv::Point2f(0, y_len - 1),
        cv::Point2f(x_len - 1, y_len - 1)
         };

    // 计算仿射变换矩阵
	Mat Affine_M = getAffineTransform(srcpoints, dstpoints);
	// cout << Affine_M << endl; // 2*3的变换矩阵

    clock_t start0 = clock();
    //对加载图形进行仿射变换操作
    Mat dst_affine;
    warpAffine(src, dst_affine, Affine_M, Size(x_len, y_len) , CV_INTER_CUBIC);
    clock_t ends0 = clock();
	cout <<"调用opencv 仿射变换库函数运行 : "<<(double)(ends0 - start0)/ CLOCKS_PER_SEC <<"s"<< endl;

	// 变换矩阵加了一行变成 3*3维的
	Mat line = (cv::Mat_<double>(1, 3) << 0,0,1);
	Affine_M.push_back(line);

	clock_t start1 = clock();
	Mat my_warpAffine_dst ;
	my_warpAffine(src,my_warpAffine_dst,x_len ,y_len, Affine_M);
	clock_t ends1 = clock();
	cout <<"自编仿射变换函数运行 : "<<(double)(ends1 - start1)/ CLOCKS_PER_SEC <<"s"<< endl;

    for (int i = 0; i < 3; i++)
    {
        circle(src, srcpoints[i], 2, Scalar(0, 0, 255), 2);
        circle(dst_affine, dstpoints[i], 2, Scalar(0, 0, 255), 2);
		circle(my_warpAffine_dst, dstpoints[i], 2, Scalar(0, 0, 255), 2);
    }

	/*###############　仿射变换　###############*/


	/*###############　透视变换　###############*/
    Point2f srcPts[4] ={ 
        cv::Point2f(345, 189),
        cv::Point2f(1128, 151),
        cv::Point2f(322, 763),
        cv::Point2f(1140, 780) 
        };
  
    Point2f dstPts[4] = {
        cv::Point2f(0, 0),
        cv::Point2f(x_len -1 , 0),
        cv::Point2f(0, y_len-1),
        cv::Point2f(x_len-1, y_len-1)
         };

    // 生成透视变换矩阵
	// 用霍夫变换
	Mat Perspective_M = getPerspectiveTransform(srcPts, dstPts);
	// cout << Perspective_M<<endl;

	clock_t start2 = clock();
	// 进行透视变换
	Mat dst_perspective;
	warpPerspective(src, dst_perspective, Perspective_M, Size(x_len, y_len) , INTER_LINEAR, BORDER_CONSTANT);
	clock_t ends2 = clock();
	cout <<"调用opencv 透视变换函数运行 : "<<(double)(ends2 - start2)/ CLOCKS_PER_SEC <<"s"<< endl;

	clock_t start3 = clock();
	Mat my_warpPerspective_dst ;
	my_warpPerspective(src,my_warpPerspective_dst,x_len ,y_len, Perspective_M);
	clock_t ends3 = clock();
	cout <<"自编透视变换函数运行 : "<<(double)(ends3 - start3)/ CLOCKS_PER_SEC <<"s"<< endl;

	clock_t start4 = clock();
	Mat my_warpPerspective_opt_dst ;
	my_warpPerspective_opt(src,my_warpPerspective_opt_dst,x_len ,y_len, Perspective_M);
	clock_t ends4 = clock();
	cout <<"自编透视变换函数优化版运行 : "<<(double)(ends4 - start4)/ CLOCKS_PER_SEC <<"s"<< endl;


    for (int i = 0; i < 4; i++)
    {
        circle(src, srcPts[i], 2, Scalar(0, 255, 0), 2);
        circle(dst_perspective, dstPts[i], 2, Scalar(0, 255, 0), 2);
		circle(my_warpPerspective_dst, dstPts[i], 2, Scalar(0, 255, 0), 2);
		circle(my_warpPerspective_opt_dst, dstPts[i], 2, Scalar(0, 255, 0), 2);
    }

	/*###############　透视变换　###############*/


	imshow("affine", dst_affine);
	imshow("perspective", dst_perspective);
	imshow("my warpAffine dst", my_warpAffine_dst);
    imshow("my warpPerspective dst", my_warpPerspective_dst);
	imshow("my warpPerspective opt dst", my_warpPerspective_opt_dst);

	imwrite("../my/affine.png", dst_affine);
	imwrite("../my/perspective.png", dst_perspective);
	imwrite("../my/my_warpAffine_dst.png", my_warpAffine_dst);
    imwrite("../my/my_warpPerspective_dst.png", my_warpPerspective_dst);
	imwrite("../my/my_warpPerspective_opt_dst.png", my_warpPerspective_opt_dst);


	waitKey();
    return 0;
} 

