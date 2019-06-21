
/* 
 220180776 胡欣毅
 数字图像处理　王桥
 课程实验 随堂版
 9周 homework 部分
 添加陈康霍夫变换部分

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

#define PI 3.14159265358979323846

//求交点
void pointXY (int point1[], int point2[], int point3[], int point4[], int point5[])
{
	double a1, b1, c1, a2, b2, c2, D;

	a1 = double(point2[1] - point1[1]);
	b1 = double(point1[0] - point2[0]);
	c1 = double(-point1[1] * (point2[0] - point1[0]) + point1[0] * (point2[1] - point1[1]));
	a2 = double(point4[1] - point3[1]);
	b2 = double(point3[0] - point4[0]);
	c2 = double(-point3[1] * (point4[0] - point3[0]) + point3[0] * (point4[1] - point3[1]));
	D = a1 * b2 - b1 * a2;
	if (D != 0) 
	{
		point5[0] = int((c1*b2 - b1 * c2) / D);
		point5[1] = int((a1*c2 - c1 * a2) / D);
	}
}


void find_point(Mat image, int cross_point[][2], int canny_low, int canny_up, int HoughLines_threshold)
{
	//边缘检测 
	Mat contours;
	int point_xy[4][2][2]; //4条直线，每条直线2个点，每点的横纵坐标
	int point_count = 0;
	cv::Canny(image, contours, canny_low, canny_up);
	vector<cv::Vec2f> lines;
	//霍夫变换,获得一组极坐标参数（rho，theta）,每一对对应一条直线，保存到lines   
	//第3,4个参数表示在（rho，theta)坐标系里横纵坐标的最小单位，即步长   
	cv::HoughLines(contours, lines, 1, PI / 180, HoughLines_threshold);
	vector<cv::Vec2f>::const_iterator it = lines.begin();
	cout << lines.size() << endl;

	while (it != lines.end()) {
		float rho = (*it)[0];
		float theta = (*it)[1];
		if (theta<PI / 4. || theta>3.*PI / 4) {
			//画交点在上下两边的直线   
			cv::Point pt1(rho / cos(theta), 0);
			cv::Point pt2((rho - image.rows*sin(theta)) / cos(theta), image.rows);
			cv::line(image, pt1, pt2, cv::Scalar(255), 1);
			point_xy[point_count][0][0] = pt1.x;
			point_xy[point_count][0][1] = pt1.y;
			point_xy[point_count][1][0] = pt2.x;
			point_xy[point_count][1][1] = pt2.y;
		}
		else {
			//画交点在左右两边的直线   
			cv::Point pt1(0, rho / sin(theta));
			cv::Point pt2(image.cols, (rho - image.cols*cos(theta) / sin(theta)));
			cv::line(image, pt1, pt2, cv::Scalar(255), 1);
			point_xy[point_count][0][0] = pt1.x;
			point_xy[point_count][0][1] = pt1.y;
			point_xy[point_count][1][0] = pt2.x;
			point_xy[point_count][1][1] = pt2.y;

		}
		++it;
		++point_count;
	}


	pointXY(point_xy[0][0], point_xy[0][1], point_xy[1][0], point_xy[1][1], cross_point[0]);
	pointXY(point_xy[0][0], point_xy[0][1], point_xy[3][0], point_xy[3][1], cross_point[1]);
	pointXY(point_xy[2][0], point_xy[2][1], point_xy[1][0], point_xy[1][1], cross_point[2]);
	pointXY(point_xy[2][0], point_xy[2][1], point_xy[3][0], point_xy[3][1], cross_point[3]);
	cout << "左下交点" << cross_point[0][0] << " " << cross_point[0][1] << endl;
	cout << "右下交点" << cross_point[1][0] << " " << cross_point[1][1] << endl;
	cout << "左上交点" << cross_point[2][0] << " " << cross_point[2][1] << endl;
	cout << "右上交点" << cross_point[3][0] << " " << cross_point[3][1] << endl;
}

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

    Mat src = cv::imread("../../ck.jpeg", cv::IMREAD_UNCHANGED);	
    // cv::imshow("src Image", src);

    Mat gray = cv::imread("../../ck.jpeg",0);
	/*###############　霍夫变换　###############*/
  
	int x_len = src.cols /8;
	int y_len = src.rows /8;

	/*###############　角点检测　###############*/
	Mat hough ;
	//若要缩小图像，一般情况下最好用CV_INTER_AREA来插值
	cv::resize(gray, hough, cv::Size(gray.cols / 8, gray.rows / 8), INTER_AREA);

	
	int cross_point[4][2];
	int canny_low = 150,canny_up = 350, HoughLines_threshold = 120 ;
	find_point(hough , cross_point, canny_low, canny_up, HoughLines_threshold);



	// 设置源图像和目标图像上的四组点以计算透射变换
	Point2f srcTri[4]={
		Point2f(cross_point[0][0], cross_point[0][1]),
		Point2f(cross_point[1][0], cross_point[1][1]),
		Point2f(cross_point[2][0], cross_point[2][1]),
		Point2f(cross_point[3][0], cross_point[3][1])
	};
	Point2f dstTri[4] = {
		Point2f(0, y_len - 1),
		Point2f(x_len - 1, y_len - 1),
		Point2f(0, 0),
		Point2f(x_len - 1, 0)
	} ;

	Mat src_resize ;
	resize(src , src_resize, Size(src.cols / 8, src.rows / 8), INTER_AREA);


	/*###############　角点检测　###############*/
  
	/*###############　仿射变换　###############*/

    // 计算仿射变换矩阵
	// 用霍夫变换
	Mat Affine_M = getAffineTransform(srcTri, dstTri);
	// cout << Affine_M << endl; // 2*3的变换矩阵

    clock_t start0 = clock();
    //对加载图形进行仿射变换操作
    Mat dst_affine;
    warpAffine(src_resize, dst_affine, Affine_M, Size(x_len, y_len) , CV_INTER_CUBIC);
    clock_t ends0 = clock();
	cout <<"调用opencv 仿射变换库函数运行 : "<<(double)(ends0 - start0)/ CLOCKS_PER_SEC <<"s"<< endl;

	// 变换矩阵加了一行变成 3*3维的
	Mat line = (cv::Mat_<double>(1, 3) << 0,0,1);
	Affine_M.push_back(line);

	clock_t start1 = clock();
	Mat my_warpAffine_dst ;
	my_warpAffine(src_resize,my_warpAffine_dst,x_len ,y_len, Affine_M);
	clock_t ends1 = clock();
	cout <<"自编仿射变换函数运行 : "<<(double)(ends1 - start1)/ CLOCKS_PER_SEC <<"s"<< endl;

    for (int i = 0; i < 3; i++)
    {
        circle(src_resize, srcTri[i], 2, Scalar(0, 0, 255), 2);
        circle(dst_affine, dstTri[i], 2, Scalar(0, 0, 255), 2);
		circle(my_warpAffine_dst, dstTri[i], 2, Scalar(0, 0, 255), 2);
    }

	/*###############　仿射变换　###############*/

	/*###############　透视变换　###############*/

    // 生成透视变换矩阵
	// 用霍夫变换
	Mat Perspective_M = getPerspectiveTransform(srcTri, dstTri);
	// cout << Perspective_M<<endl;

	clock_t start2 = clock();
	// 进行透视变换
	Mat dst_perspective;
	warpPerspective(src_resize, dst_perspective, Perspective_M, Size(x_len, y_len) , INTER_LINEAR, BORDER_CONSTANT);
	clock_t ends2 = clock();
	cout <<"调用opencv 透视变换函数运行 : "<<(double)(ends2 - start2)/ CLOCKS_PER_SEC <<"s"<< endl;

	clock_t start3 = clock();
	Mat my_warpPerspective_dst ;
	my_warpPerspective(src_resize,my_warpPerspective_dst,x_len ,y_len, Perspective_M);
	clock_t ends3 = clock();
	cout <<"自编透视变换函数运行 : "<<(double)(ends3 - start3)/ CLOCKS_PER_SEC <<"s"<< endl;

	clock_t start4 = clock();
	Mat my_warpPerspective_opt_dst ;
	my_warpPerspective_opt(src_resize,my_warpPerspective_opt_dst,x_len ,y_len, Perspective_M);
	clock_t ends4 = clock();
	cout <<"自编透视变换函数优化版运行 : "<<(double)(ends4 - start4)/ CLOCKS_PER_SEC <<"s"<< endl;


    for (int i = 0; i < 4; i++)
    {
        circle(src_resize, srcTri[i], 2, Scalar(0, 255, 0), 2);
        circle(dst_perspective, dstTri[i], 2, Scalar(0, 255, 0), 2);
		circle(my_warpPerspective_dst, dstTri[i], 2, Scalar(0, 255, 0), 2);
		circle(my_warpPerspective_opt_dst, dstTri[i], 2, Scalar(0, 255, 0), 2);
    }


	/*###############　透视变换　###############*/

	imshow("hough", hough);
	imshow("affine", dst_affine);
	imshow("perspective", dst_perspective);
	imshow("my warpAffine dst", my_warpAffine_dst);
    imshow("my warpPerspective dst", my_warpPerspective_dst);
	imshow("my warpPerspective opt dst", my_warpPerspective_opt_dst);

	imwrite("../ck/hough.png", hough);
	imwrite("../ck/affine.png", dst_affine);
	imwrite("../ck/perspective.png", dst_perspective);
	imwrite("../ck/my_warpAffine_dst.png", my_warpAffine_dst);
    imwrite("../ck/my_warpPerspective_dst.png", my_warpPerspective_dst);
	imwrite("../ck/my_warpPerspective_opt_dst.png", my_warpPerspective_opt_dst);


	waitKey();
    return 0;
} 

