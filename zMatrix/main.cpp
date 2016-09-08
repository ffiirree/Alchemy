#include <iostream>

#include"config_default.h"
#include "zmatrix.h"
#include "zimgproc.h"
#include <opencv2\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <string>

using namespace std;
using namespace cv;
using namespace z;


int main(int argc, char *argv[])
{
	Mat gray, display, src = imread("test.jpeg");
	Matrix8u mgray, m = Mat2Matrix8u(src);
	

	cvtColor(src, gray, CV_BGR2GRAY);
	mgray = Mat2Matrix8u(gray);

	imshow("org", Mat(m));
	imshow("gray", gray);

	cv::medianBlur(src, display, 5);
	imshow("caitu", display);

	
	m = z::medianFilter(m, z::Size(5, 5));
	imshow("d", Mat(m)); 

	waitKey(0);
	return 0;
}