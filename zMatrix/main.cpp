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

	imshow("zOrg", Mat(m));

	timestamp.start();
	z::GaussianBlur(m, mdis, z::Size(5, 5));
	timestamp.runtime();
	imshow("zGuss", Mat(mdis));

	timestamp.start();
	z::blur(m, mdis, z::Size(5, 5));
	timestamp.runtime();
	imshow("zBlur", Mat(mdis));


	timestamp.start();
	z::medianFilter(m, mdis, z::Size(5, 5));
	timestamp.runtime();
	imshow("zMedia", Mat(mdis));

	waitKey(0);
	return 0;
}