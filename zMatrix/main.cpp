#include <iostream>
#include <opencv2\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <string>
#include <ctime>  

#include "zcore.h"
#include "zimgproc\zimgproc.h"
#include "zimgproc\transform.h"
#include "zgui\zgui.h"
#include "debug.h"

using namespace std;
using namespace z;


int main(int argc, char *argv[])
{
	Matrix8u zcolor, zgray, zdis;
	cv::Mat cvColor, cvGray, cvdis;
	TimeStamp timestamp;

	// 读取一张图片
	zcolor = imread("test.jpeg");

	// 显示图片需要用openCV的函数
	cv::imshow("zcolor", cv::Mat(zcolor));

	// 转化为灰度图像
	cvtColor(zcolor, zgray, BGR2GRAY);
	cv::imshow("zgray", cv::Mat(zgray));

	// 高斯滤波
	GaussianBlur(zcolor, zdis, Size(5, 5));
	cv::imshow("z GassianBlar", cv::Mat(zdis));

	// sobel 一阶微分算子边缘检测
	sobel(zgray, zdis, 1, 1, 3);
	cv::imshow("z sobel", cv::Mat(zdis));

 	cv::waitKey(0);
	return 0;
}