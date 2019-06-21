/* 
 220180776 胡欣毅
 数字图像处理　王桥
 课程实验四

 */

#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define PI 3.14159265358979323846

double sigmoid(double x)
{
    return 1. / (1. + exp(-x) );
}

double step_function(double x)
{
    if (x > 0) return 1.0;
    else       return 0.0;
}

double gaosi(double sigma = 1.0,double  x = 0)
{
    return exp(-pow(x,2)/(2*pow(sigma,2)) )/(sqrt(2*PI)*sigma);
}

// 基于数组

void linspace(double *f , double begin, double finish, int number) 
{
	double interval = (finish - begin) / (number - 1);

    for (int i = 0; i < number; i++) 
    {
        f[i]= begin + i * interval;
    }
	
}

void diff(double * x , double * ddy, int number)
{  
    double dy[number-1];
    for(int i = 0; i < number-1; i++)
    {
        dy[i] = x[i+1] - x[i];
    }

    for(int i = 0; i < number-2; i++)
    {
        ddy[i] = dy[i+1] - dy[i];
    }

}

void conv(double *x , double *y ,int x_len, int y_len, double * out)
{
    int out_len = x_len + y_len -1;
    
    // x　反转
    double xx[x_len];
    for (int i = 0;i < x_len ; i++)
    {
        xx[i] = x[x_len -1 - i];
    }

    double tmp[2*x_len+y_len-2];
    for (int i =0;i<2*x_len+y_len-2 ; i++)
    {
        if (i<x_len-1)            tmp[i] = 0;
        else if (i<x_len+y_len-1) tmp[i] = y[i-x_len+1];
        else                      tmp[i]=0;
    }

    for (int i =0;i<out_len;i++)
    {   
        double sum = 0;
        for (int j =0;j<x_len;j++)
        {
            sum += xx[j]*tmp[i+j];
        }
        out[i] = sum;
    }
}


int main() 
{    
    // 一维数据增强　part 1
    // x
    const int num = 200 ;
    double x[num] ;
    linspace(x, -10.0, 10.0 , num);
    //for(int i = 0;i<num;i++) cout<<x[i]<<endl;

    // ddy
    double ddy[num-2];
    diff(x,ddy,num);
    //for(int i = 0;i<num-2 ; i++) cout<<ddy[i]<<endl;

    // 一维数据增强
    double lamb = 400;
    double out[num-2];
    for(int i = 0;i < num-2 ; i++)  
    {
        out[i] = x[i+1] - lamb * ddy[i];
        // cout<<out[i]<<endl;
    }

/*
    test conv
    double xx[4] = {1,2,3,4};
    double y[5] = {4,5,6,7,8};
    double newout[8] ;
    int x_len = sizeof(xx) / sizeof(xx[0]);
    int y_len = sizeof(y) / sizeof(y[0]);
    conv(xx,y,x_len,y_len,newout);
    for(int i = 0;i<8;i++) cout<<newout[i]<<endl;
    //4    13    28    50    60    61    52    32
    
*/

    // 一维数据柔化增强　part 2
    // 阶跃函数 y
    double y[num];
    for(int i = 0;i<num;i++) y[i] = step_function(x[i]);
    // for(int i = 0;i<num;i++) cout<<y[i]<<endl;
    
    // 阶跃函数 y 的柔化
    // 高斯滤波器
    double sigma = 2.0;
    double tmp[(int) (8*sigma) ];
    linspace(tmp , -3*sigma, 3*sigma, (int) (8*sigma) );

    double gaosi_filter[(int) (8*sigma)];
    for(int i = 0;i<num;i++) gaosi_filter[i] = gaosi(sigma,tmp[i]);

    // y 与高斯 卷积
    int x_len = sizeof(y) / sizeof(y[0]);
    int y_len = sizeof(gaosi_filter) / sizeof(gaosi_filter[0]);
    
    double conv_out[x_len+y_len-1];
    int conv_len = sizeof(conv_out) / sizeof(conv_out[0]);

    conv(y , gaosi_filter ,x_len,  y_len,  conv_out);
    // 一维数据柔化 finish
    
    double ddout[conv_len -2 ];
    diff(conv_out,ddout,conv_len);

    // 一维数据增强
    double lambd = 10;
    double strengthen[conv_len-2];
    for(int i = 0;i < conv_len-2 ; i++)  
    {
        strengthen[i] = conv_out[i+1] - lambd * ddout[i];
        //cout<<strengthen[i]<<endl;
    }


    return 0;  
} 

