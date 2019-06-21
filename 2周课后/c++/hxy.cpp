/* 
 220180776 胡欣毅
 数字图像处理　王桥
 课程实验一

 */

#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat RotateImage(Mat& src, double angle,double scale = 1.0)
{  	    	
	//输出图像的尺寸与原图一样    
	cv::Size dst_sz(src.cols, src.rows);
	//指定旋转中心      
	Point2f center(static_cast<float>(src.cols / 2.), static_cast<float>(src.rows / 2.));
	//获取旋转矩阵（2x3矩阵）      
	Mat rot_mat = cv::getRotationMatrix2D(center, angle, scale);
	//复制边缘填充
	cv::Mat dst; 
	cv::warpAffine(src, dst, rot_mat, dst_sz); 
	return dst;
}

Mat togray(Mat& src)
{
	// Hint: Gray=R×0.299+G×0.587+B×0.114  
	vector<Mat> channels;
	split(src, channels);
	Mat B_c = channels.at(0);
	Mat G_c = channels.at(1);
	Mat R_c =channels.at(2);

	Mat Gray = R_c * 0.299 + G_c * 0.587 + B_c * 0.114;
	return Gray ;
}

	
int main()
{
	Mat im = cv::imread("../../4.2.03.tiff", -1 );	
	cv::imshow("原始图", im);

	// 2. 图像像素级读写：求横坐标为25的所有像素三个通道像素值之和；（10/100）
	cout<<"横坐标为25的所有像素三个通道像素值之和: "<<sum(sum(im.row(25)))[0]<<endl;

	// 3. 利用公式将RGB图像转化为灰度图，并显示；（20/100）
	Mat im_Gray = togray(im) ; 

	cv::imshow("灰度图", im_Gray);
	cv::imwrite("../im_Gray.png", im_Gray); //保存图像

	// 4. 对题目3得到的灰度图像进行二值化，给定上界A与下界B，在[A, B]内的像素值设置为255，反之为0；（20/100）

	float A = 100;
	float B = 180;

	Mat im_Binarization = im_Gray >= A & im_Gray <= B;

/*
	Mat im_Binarization =  cv::Mat::zeros(im_Gray.rows, im_Gray.cols, CV_8UC1); //灰度图初始 ;
	for(int i = 0 ;i<im_Gray.rows ;i++)
	{
		for(int j = 0 ;j<im_Gray.cols ;j++)
		{
			if (im_Gray.at<uchar>(i,j)>=A & im_Gray.at<uchar>(i,j)<=B)
			{
				im_Binarization.at<uchar>(i,j) = 255;
			}
			else
			{
				im_Binarization.at<uchar>(i,j) = 0;
			}
			
		}
	}
*/

	cv::imshow("二值化图", im_Binarization);	
	cv::imwrite("../im_Binarization.png", im_Binarization); //保存图像

	// 5. 图像顺时针旋转30度，并显示；（30/100）

	cv::Mat im_rotate = RotateImage(im,-30, 0.70);
	
	cv::imshow("旋转图", im_rotate);
	cv::imwrite("../im_rotate.png", im_rotate); //保存图像

	// 6. 在图像中心截取像素为256*256的子图，并显示；（20/100）

    // int srcHeight, srcWidth, subHeight, subWidth;

	cv::Rect m_select;
	m_select = Rect(im.rows/4,im.cols/4,256,256);
	Mat im_child = im(m_select);
 
	cv::imshow("子图", im_child);
	cv::imwrite("../im_child.png", im_child); //保存图像

    waitKey();
    return 0;
}


	
