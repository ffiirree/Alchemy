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

using namespace std;
using namespace z;


int main(int argc, char *argv[])
{
	Matrix8u img = imread("test.jpeg");
	cv::imshow("read", cv::Mat(img));

	Matrix8u gray;
	cvtColor(img, gray, BGR2GRAY);
	cv::imshow("gray", cv::Mat(gray));

 	cv::waitKey(0);
	return 0;
}