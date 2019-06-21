

#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>
#include<iostream>
#include<math.h>

using namespace std;
using namespace cv;

//小波分解
void laplace_decompose(Mat src,  Mat &wave, int s)
{
	Mat full_src(src.rows, src.cols, CV_32FC1);
	Mat dst = src.clone();
	dst.convertTo(dst, CV_32FC1);

	for (int m = 0; m < s; m++)
	{
		dst.convertTo(dst, CV_32FC1);
		Mat wave_src(dst.rows, dst.cols, CV_32FC1);
		//列变换，detail=mean-original
		for (int i = 0; i < wave_src.rows; i++)
		{
			for (int j = 0; j < wave_src.cols / 2; j++)
			{
				wave_src.at<float>(i, j) = (dst.at<float>(i, 2 * j) + dst.at<float>(i, 2 * j + 1)) / 2;
				wave_src.at<float>(i, j + wave_src.cols / 2) = wave_src.at<float>(i, j) - dst.at<float>(i, 2 * j);
			}
		}
		Mat temp = wave_src.clone();
		for (int i = 0; i < wave_src.rows / 2; i++)
		{
			for (int j = 0; j < wave_src.cols / 2; j++)
			{
				wave_src.at<float>(i, j) = (temp.at<float>(2 * i, j) + temp.at<float>(2 * i + 1, j)) / 2;
				wave_src.at<float>(i + wave_src.rows / 2, j) = wave_src.at<float>(i, j) - temp.at<float>(2 * i, j);
			}
		}
		dst.release();
		dst = wave_src(Rect(0, 0, wave_src.cols / 2, wave_src.rows / 2));
		wave_src.copyTo(full_src(Rect(0, 0, wave_src.cols, wave_src.rows)));
	}
	wave = full_src.clone();
}
//小波复原
void wave_recover(Mat full_scale, Mat &original, int level)
{
	//每一个循环中把一个级数的小波还原
	for (int m = 0; m < level; m++)
	{
		Mat temp = full_scale(Rect(0, 0, full_scale.cols / pow(2, level - m - 1), full_scale.rows / pow(2, level - m - 1)));

		//先恢复左边
		Mat recover_src(temp.rows, temp.cols, CV_32FC1);
		for (int i = 0; i < recover_src.rows; i++)
		{
			for (int j = 0; j < recover_src.cols / 2; j++)
			{
				if (i % 2 == 0)
					recover_src.at<float>(i, j) = temp.at <float>(i / 2, j) - temp.at<float>(i / 2 + recover_src.rows / 2, j);
				else
					recover_src.at<float>(i, j) = temp.at <float>(i / 2, j) + temp.at<float>(i / 2 + recover_src.rows / 2, j);
			}
		}
		Mat temp2 = recover_src.clone();
		//再恢复整个
		for (int i = 0; i < recover_src.rows; i++)
		{
			for (int j = 0; j < recover_src.cols; j++)
			{
				if (j % 2 == 0)
					recover_src.at<float>(i, j) = temp2.at<float>(i, j / 2) - temp.at<float>(i, j / 2 + temp.cols / 2);
				else
					recover_src.at<float>(i, j) = temp2.at<float>(i, j / 2) + temp.at<float>(i, j / 2 + temp.cols / 2);
			}
		}
		recover_src.copyTo(temp);
	}
	original = full_scale.clone();
	original.convertTo(original, CV_8UC1);


}

//小波操作
void ware_operate(Mat &full_scale, int level)
{
	//取出最低尺度的那一层，对其进行操作，仅最低尺度那层可以对时域进行操作，其他层只能对频域进行操作
	Mat temp = full_scale(Rect(0, 0, full_scale.cols / 4, full_scale.rows / 4));
	Mat temp1 = temp(Rect(0, 0, temp.cols / 2, temp.rows / 2));
	Mat temp2 = temp1.clone();
	//这里对时域操作，降低灰度
	for (int i = 0; i < temp2.rows; i++)
		for (int j = 0; j < temp2.cols; j++)
			temp2.at<float>(i, j) -= 20;
	temp2.copyTo(temp1);

	//这里对频域操作，拉伸细节
	//先处理左下角
	for (int i = temp.rows / 2; i < temp.rows; i++)
	{
		for (int j = 0; j < temp.cols / 2; j++)
		{
			if (temp.at<float>(i, j) > 0)
				temp.at<float>(i, j) += 5;
			if (temp.at<float>(i, j) < 0)
				temp.at<float>(i, j) -= 5;
		}
	}
	//再处理右半边
	for (int i = 0; i < temp.rows; i++)
	{
		for (int j = temp.cols / 2; j < temp.cols; j++)
		{
			if (temp.at<float>(i, j) > 0)
				temp.at<float>(i, j) += 5;
			if (temp.at<float>(i, j) < 0)
				temp.at<float>(i, j) -= 5;
		}
	}
}

//小波分解
Mat waveletDecompose(const Mat &_src, const Mat &_lowFilter, const Mat &_highFilter)
{
	assert(_src.rows == 1 && _lowFilter.rows == 1 && _highFilter.rows == 1);
	assert(_src.cols >= _lowFilter.cols && _src.cols >= _highFilter.cols);
	Mat src = Mat_<float>(_src);

	int D = src.cols;

	Mat lowFilter = Mat_<float>(_lowFilter);
	Mat highFilter = Mat_<float>(_highFilter);

	//频域滤波或时域卷积；ifft( fft(x) * fft(filter)) = cov(x,filter) 
	Mat dst1 = Mat::zeros(1, D, src.type());
	Mat dst2 = Mat::zeros(1, D, src.type());

	filter2D(src, dst1, -1, lowFilter);
	filter2D(src, dst2, -1, highFilter);

	//下采样
	Mat downDst1 = Mat::zeros(1, D / 2, src.type());
	Mat downDst2 = Mat::zeros(1, D / 2, src.type());

	resize(dst1, downDst1, downDst1.size());
	resize(dst2, downDst2, downDst2.size());

	//数据拼接
	for (int i = 0; i < D / 2; i++)
	{
		src.at<float>(0, i) = downDst1.at<float>(0, i);
		src.at<float>(0, i + D / 2) = downDst2.at<float>(0, i);

	}
	return src;
}


int main() //int, char** argv )  
{   
/*
    cv::Mat src = imread( argv[1], cv::IMREAD_UNCHANGED);
    if( !src.data )
    { return -1; }
*/
    Mat im = imread("../../lena_std.tif" , -1);
    //imshow("原始图像", im);
    
    Mat im_gray;
    cvtColor(im, im_gray, CV_RGB2GRAY);
    imshow("gray", im_gray);

    im_gray = Mat_<float>(im_gray) ;

	Mat dwt ,tmp;
	laplace_decompose(im_gray, dwt , 3);
	ware_operate(dwt, 3);

	normalize(dwt,tmp,0,1,CV_MINMAX);
	//dwt.convertTo(tmp, CV_8UC1, 255, 0);
	imshow("dwt", tmp);

	Mat idwt;
	wave_recover(dwt, idwt, 3);

    //normalize(idwt,idwt,0,1,CV_MINMAX);
	//idwt.convertTo(idwt, CV_8UC1, 255, 0);
	imshow("idwt", idwt);

	waitKey();
	return 0 ;
}


