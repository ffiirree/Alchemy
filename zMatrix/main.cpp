#include <iostream>

#include "zcore.h"
#include "zimgproc\zimgproc.h"
#include "zimgproc\transform.h"
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
	Matrix8u mgray, mdis, mcolor = Mat2Matrix8u(src);

	// ‘≠Õº
	imshow("orginal", Mat(mcolor));

	// ª“∂»Õº
	z::cvtColor(mcolor, mgray, BGR2GRAY);
	imshow("gray", Mat(mgray));

	// sobel ≤ Õº±ﬂ‘µºÏ≤‚
	z::sobel(mcolor, mdis);
	imshow("sobel", Mat(mdis));

	cv::Sobel(src, display, CV_8U, 1, 1);
	imshow("cvsobel", display);

	// canny ª“∂»Õº±ﬂ‘µºÏ≤‚
	z::Canny(mgray, mdis, 30, 90);
	imshow("canny gray", Mat(mdis));

	cv::Canny(src, display, 30, 90, 3);
	imshow("cvcanny gray", display);

 	waitKey(0);
	return 0;
}