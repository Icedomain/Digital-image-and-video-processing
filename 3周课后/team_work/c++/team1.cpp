/* 
 220180776 胡欣毅
 数字图像处理　王桥
 课程实验(小组作业1)

 */
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
using namespace cv;
using namespace std;
 
//给原图像增加椒盐噪声
//图象模拟添加椒盐噪声是通过随机获取像素点斌那个设置为高亮度点来实现的
Mat addSaltNoise(const Mat srcImage, int n)
{
	Mat dstImage = srcImage.clone();
	for (int k = 0; k < n; k++)
	{
		//随机取值行列
		int i = rand() % dstImage.rows;
		int j = rand() % dstImage.cols;
		//图像通道判定
		if (dstImage.channels() == 1)
		{
			dstImage.at<uchar>(i, j) = 255;		//盐噪声
		}
		else
		{
			dstImage.at<Vec3b>(i, j)[0] = 255;
			dstImage.at<Vec3b>(i, j)[1] = 255;
			dstImage.at<Vec3b>(i, j)[2] = 255;
		}
	}
	for (int k = 0; k < n; k++)
	{
		//随机取值行列
		int i = rand() % dstImage.rows;
		int j = rand() % dstImage.cols;
		//图像通道判定
		if (dstImage.channels() == 1)
		{
			dstImage.at<uchar>(i, j) = 0;		//椒噪声
		}
		else
		{
			dstImage.at<Vec3b>(i, j)[0] = 0;
			dstImage.at<Vec3b>(i, j)[1] = 0;
			dstImage.at<Vec3b>(i, j)[2] = 0;
		}
	}
	return dstImage;
}

Mat RotateImage(Mat src, double angle,float scale = 1.0 )
{
	Mat dst;   	    	
	//输出图像的尺寸与原图一样    
	cv::Size dst_sz(src.cols, src.rows);
	//指定旋转中心      
	cv::Point2f center(static_cast<float>(src.cols / 2.), static_cast<float>(src.rows / 2.));
	//获取旋转矩阵（2x3矩阵）      
	Mat rot_mat = cv::getRotationMatrix2D(center, angle, scale);
	//复制边缘填充
	cv::warpAffine(src, dst, rot_mat, dst_sz); 

	return dst;
}

int main()
{
    //Create SIFT or SURF class pointer
    
    Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
    // Ptr<Feature2D> f2d = xfeatures2d::SURF::create();


    //读入图片
    Mat img_1 = imread("../../../data/4.1.05.tiff");

    // img_1 = addSaltNoise(img_1, 300);
	// imshow("添加椒盐噪声的图像", img_1);

    double angle = -45;
    float scale = 0.7;
	Mat img_2 =  RotateImage(img_1,angle , scale);
   
    //Detect the keypoints
    vector<KeyPoint> keypoints_1, keypoints_2;
    f2d->detect(img_1, keypoints_1);
    f2d->detect(img_2, keypoints_2);

    //绘制特征点(关键点)
    Mat feature_pic1, feature_pic2;
    drawKeypoints(img_1, keypoints_1, feature_pic1, Scalar::all(-1));
    drawKeypoints(img_2, keypoints_2, feature_pic2, Scalar::all(-1));
    imshow("img_1_feature", feature_pic1);
    imshow("img_2_feature", feature_pic2);
    imwrite("../img_1_feature.png", feature_pic1);
    imwrite("../img_2_feature.png", feature_pic2);

    //Calculate descriptors (feature vectors)
    Mat descriptors_1, descriptors_2;
    f2d->compute(img_1, keypoints_1, descriptors_1);
    f2d->compute(img_2, keypoints_2, descriptors_2);  

    // 用BFMatcher进行匹配

    BFMatcher matcher;
    vector<DMatch> matches;
    matcher.match(descriptors_1, descriptors_2, matches); //实现描述符之间的匹配
    
    /*
    //提取出前30个最佳匹配结果  
    nth_element(matches.begin(), matches.begin()+29, matches.end());      
    matches.erase(matches.begin()+30, matches.end());    //剔除掉其余的匹配结果
    */
    //绘制匹配出的关键点
    Mat img_matches;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_matches);
    imshow("初步匹配图", img_matches);
    imwrite("../match1.png",img_matches);


    //计算向量距离的最大值与最小值：距离越小越匹配
    double max_dist=matches[0].distance;
    double min_dist=matches[0].distance;

    for(int i=1; i<descriptors_1.rows; i++)
    {
        if(matches.at(i).distance > max_dist)
            max_dist = matches[i].distance;
        if(matches.at(i).distance < min_dist)
            min_dist = matches[i].distance;
    }
    cout<<"min_distance="<<min_dist<<endl;
    cout<<"max_distance="<<max_dist<<endl;

    //匹配结果删选    
    vector<DMatch>good_matches;
    for(int i=0; i<matches.size(); i++)
    {
        if(matches[i].distance < 2 * min_dist)
            good_matches.push_back(matches[i]);
    }

    Mat result;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, result,  Scalar(0, 255, 0), Scalar::all(-1));
    imshow("修正匹配图", result);
    imwrite("../match2.png",result);

    waitKey();
    return 0;
}



