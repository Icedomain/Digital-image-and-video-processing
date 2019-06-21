/* 
 220180776 胡欣毅
 数字图像处理　王桥
 Wavelet

 */


// https://blog.csdn.net/songyimin1208/article/details/52717433?utm_source=blogxgwz3
// https://blog.csdn.net/roslei/article/details/73459418

#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>
#include<iostream>
#include<math.h>

using namespace cv;
using namespace std;


// 生成不同类型的小波，现在只有haar，sym2
void wavelet( string &_wname, Mat &_lowFilter, Mat &_highFilter)
{
    if (_wname == "sym2")
    {
        int   N   = 4;
        float h[] = { -0.483, 0.836, -0.224, -0.129 };
        float l[] = { -0.129, 0.224, 0.837, 0.483 };


        _lowFilter  = Mat::zeros(1, N, CV_32F);
        _highFilter = Mat::zeros(1, N, CV_32F);


        for (int i = 0; i<N; i++)
        {
        _lowFilter.at<float>  (0, i) = l[i];
        _highFilter.at<float>(0, i)  = h[i];
        }

    }
        
    if ( _wname=="haar" || _wname=="db1" )
    {
        int N = 2;
        _lowFilter = Mat::zeros( 1, N, CV_32F );
        _highFilter = Mat::zeros( 1, N, CV_32F );
        
        _lowFilter.at<float>(0, 0) = 1/sqrtf(N); 
        _lowFilter.at<float>(0, 1) = 1/sqrtf(N); 
 
        _highFilter.at<float>(0, 0) = -1/sqrtf(N); 
        _highFilter.at<float>(0, 1) = 1/sqrtf(N); 
    }
}


// 小波分解
Mat waveletDecompose( Mat _src,  Mat &_lowFilter,  Mat &_highFilter)
{
    assert(_src.rows == 1 && _lowFilter.rows == 1 && _highFilter.rows == 1);
    assert(_src.cols >= _lowFilter.cols && _src.cols >= _highFilter.cols);
   
    Mat src = Mat_<float>(_src);
    int D = src.cols;

    Mat lowFilter  = Mat_<float>(_lowFilter);
    Mat highFilter = Mat_<float>(_highFilter);

    /// 频域滤波，或时域卷积；ifft( fft(x) * fft(filter)) = cov(x,filter) 
    Mat dst1 = Mat::zeros(1, D, src.type());
    Mat dst2 = Mat::zeros(1, D, src.type());

    filter2D(src, dst1, -1, lowFilter);
    filter2D(src, dst2, -1, highFilter);

    /// 下采样
    Mat downDst1 = Mat::zeros(1, D / 2, src.type());
    Mat downDst2 = Mat::zeros(1, D / 2, src.type());


    resize(dst1, downDst1, downDst1.size());
    resize(dst2, downDst2, downDst2.size());

    /// 数据拼接
    for (int i = 0; i<D / 2; i++)
    {
        src.at<float>(0, i)         = downDst1.at<float>(0, i);
        src.at<float>(0, i + D / 2) = downDst2.at<float>(0, i);
    }

    return src;
}


// 小波重建
Mat waveletReruct( Mat _src,  Mat &_lowFilter,  Mat &_highFilter)
{
    assert(_src.rows == 1 && _lowFilter.rows == 1 && _highFilter.rows == 1);
    assert(_src.cols >= _lowFilter.cols && _src.cols >= _highFilter.cols);
    
    Mat src = Mat_<float>(_src);
    int D = src.cols;

    Mat lowFilter  = Mat_<float>(_lowFilter);
    Mat highFilter = Mat_<float>(_highFilter);

    /// 插值;
    Mat Up1 = Mat::zeros(1, D, src.type());
    Mat Up2 = Mat::zeros(1, D, src.type());

    // 插值为0
    // for ( int i=0, cnt=1; i<D/2; i++,cnt+=2 )
    // {
    //     Up1.at<float>( 0, cnt ) = src.at<float>( 0, i );     ///< 前一半
    //     Up2.at<float>( 0, cnt ) = src.at<float>( 0, i+D/2 ); ///< 后一半
    // }

    /// 线性插值
    Mat roi1(src, Rect(0, 0, D / 2, 1));
    Mat roi2(src, Rect(D / 2, 0, D / 2, 1));
    resize(roi1, Up1, Up1.size(), 0, 0, INTER_CUBIC);
    resize(roi2, Up2, Up2.size(), 0, 0, INTER_CUBIC);


    /// 前一半低通，后一半高通
    Mat dst1 = Mat::zeros(1, D, src.type());
    Mat dst2 = Mat::zeros(1, D, src.type());
    filter2D(Up1, dst1, -1, lowFilter);
    filter2D(Up2, dst2, -1, highFilter);

    /// 结果相加
    dst1 = dst1 + dst2;
    return dst1;
} 


///  小波变换
Mat WDT( Mat &_src,  string _wname,  int _level)
{
    //int reValue = THID_ERR_NONE;
    Mat src = Mat_<float>(_src);
    Mat dst = Mat::zeros(src.rows, src.cols, src.type());
    int N   = src.rows;
    int D   = src.cols;


    /// 高通低通滤波器
    Mat lowFilter;
    Mat highFilter;
    wavelet(_wname, lowFilter, highFilter);


    /// 小波变换
    int t   = 1;
    int row = N;
    int col = D;


    while (t <= _level)
    {
        ///先进行行小波变换
        for (int i = 0; i<row; i++)
        {
            /// 取出src中要处理的数据的一行
            Mat oneRow = Mat::zeros(1, col, src.type());
            for (int j = 0; j<col; j++){
                oneRow.at<float>(0, j) = src.at<float>(i, j);
            }
            oneRow = waveletDecompose(oneRow, lowFilter, highFilter);
            /// 将src这一行置为oneRow中的数据
            for (int j = 0; j<col; j++){
                dst.at<float>(i, j) = oneRow.at<float>(0, j);
            }
        }

        /// 小波列变换
        for (int j = 0; j<col; j++){
            /// 取出src数据的一行输入
            Mat oneCol = Mat::zeros(row, 1, src.type());
            for (int i = 0; i<row; i++){
                oneCol.at<float>(i, 0) = dst.at<float>(i, j);
            }
            oneCol = ( waveletDecompose(oneCol.t(), lowFilter, highFilter)).t();

            for (int i = 0; i<row; i++){
            dst.at<float>(i, j) = oneCol.at<float>(i, 0);
            }
        }

        /// 更新
        row /= 2;
        col /= 2;
        t++;
        src = dst;
    }

    return dst;
}


///  小波逆变换
Mat IWDT( Mat &_src,  string _wname,  int _level)
{
    //int reValue = THID_ERR_NONE;
    Mat src = Mat_<float>(_src);
    Mat dst = Mat::zeros(src.rows, src.cols, src.type());
    int N   = src.rows;
    int D   = src.cols;

    /// 高通低通滤波器
    Mat lowFilter;
    Mat highFilter;
    wavelet(_wname, lowFilter, highFilter);


    /// 小波变换
    int t   = 1;
    int row = N / pow(2., _level - 1);
    int col = D / pow(2., _level - 1);


    while (row <= N && col <= D)
    {
        /// 小波列逆变换
        for (int j = 0; j<col; j++)
        {
            /// 取出src数据的一行输入
            Mat oneCol = Mat::zeros(row, 1, src.type());
            for (int i = 0; i<row; i++){
                oneCol.at<float>(i, 0) = src.at<float>(i, j);
            }
            oneCol = waveletReruct(oneCol.t(), lowFilter, highFilter).t();


            for (int i = 0; i<row; i++){
                dst.at<float>(i, j) = oneCol.at<float>(i, 0);
            }
        }


        ///行小波逆变换
        for (int i = 0; i<row; i++){
            /// 取出src中要处理的数据的一行
            Mat oneRow = Mat::zeros(1, col, src.type());
            for (int j = 0; j<col; j++){
                oneRow.at<float>(0, j) = dst.at<float>(i, j);
            }
            oneRow = waveletReruct(oneRow, lowFilter, highFilter);
            /// 将src这一行置为oneRow中的数据
            for (int j = 0; j<col; j++){
                dst.at<float>(i, j) = oneRow.at<float>(0, j);
            }
        }

    row *= 2;
    col *= 2;
    src  = dst;

    }

    return dst;
}


int main() //int, char** argv )  
{   
/*
    cv::Mat src = imread( argv[1], cv::IMREAD_UNCHANGED);
    if( !src.data )
    { return -1; }
*/
    Mat im = imread("../../lena_std.tif");
    //imshow("原始图像", im);
    int height = im.rows;
    int width  = im.cols;
    
    Mat im_gray;
    cvtColor(im, im_gray, CV_RGB2GRAY);
    imshow("gray", im_gray);

    im_gray = Mat_<float>(im_gray) ;

    string wname = "haar";
    int depth ;
    cout << "输入深度："<<endl;
    cin >> depth ;
    Mat im_wdt = WDT(im_gray, wname, depth);
    Mat im_iwdt = IWDT(im_wdt, wname, depth);

    normalize(im_wdt,im_wdt,0,1,CV_MINMAX);
    normalize(im_iwdt,im_iwdt,0,1,CV_MINMAX);
    imshow("wdt", im_wdt);
    imshow("iwdt", im_iwdt);

    im_wdt.convertTo(im_wdt, CV_8UC1, 255, 0);
    im_iwdt.convertTo(im_iwdt, CV_8UC1, 255, 0);
    
    // cout << im_wdt.rowRange(1,2);

    waitKey();  
    return 0;  
} 



