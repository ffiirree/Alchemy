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
	Matrix8u gray;
	cvtColor(img, gray, BGR2GRAY);

	cv::Mat test = gray;
	cv::Mat pa[] = { cv::Mat_<double>(test), cv::Mat::zeros(test.size(), CV_64F) };
	cv::Mat com;

	cv::merge(pa, 2, com);
	cv::dft(com, com);

	Matrix dst;
	dft(gray, dst);


	cv::waitKey(0);
	return 0;
}


