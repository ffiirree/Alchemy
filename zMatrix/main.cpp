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
	Matrix8u img = imread("test.jpeg");

	line(img, Point(10, 20), Point(100, 68), Scalar(255, 255, 255));

	// 显示彩色图片
	imshow("hello", img);
	waitKey(0);

	//cv::imshow("hello", cv::Mat(img));
	//cv::waitKey(0);
	return 0;
}


