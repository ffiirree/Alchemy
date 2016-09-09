#include <iostream>

#include"config_default.h"
#include "zmatrix.h"
#include "zimgproc.h"
#include <opencv2\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <string>

#include <ctime>  

using namespace std;
using namespace cv;
using namespace z;


int main(int argc, char *argv[])
{
	TimeStamp timestamp;
	Mat gray, display, src;
	src = imread("test.jpeg");
	Matrix8u mgray, mdis, m = Mat2Matrix8u(src);

	// openCV
	timestamp.start();
	cv::blur(src, display, cv::Size(5, 5));
	timestamp.runtime();
	imshow("cv", display);

	// zMatrix
	timestamp.start();
	z::blur(m, mdis, z::Size(5, 5));
	timestamp.runtime();
	imshow("z", Mat(mdis));

	waitKey(0);
	return 0;
}