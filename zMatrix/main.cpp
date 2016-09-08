#include <iostream>

#include"config_default.h"
#include "zmatrix.h"
#include "zimgproc.h"
#include <opencv2\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace std;
using namespace cv;
using namespace z;

int main(int argc, char *argv[])
{
	Mat gray, display, src = imread("test.jpeg");
	cvtColor(src, gray, CV_BGR2GRAY);

	//imshow("src", src);
	imshow("gray", gray);

	Matrix8u m = Mat2Matrix8u(gray);

	Mat mida = gray.clone();

	cv::medianBlur(gray, mida, (7, 7));
	imshow("mida", mida);

	//display = z::blur(m, z::Size(3, 3));
	//imshow("filter3", display);

	//display = z::blur(m, z::Size(5, 5));
	//imshow("filter5", display);

	display = medianFilter(m, z::Size(7, 7));
	imshow("emboss", display);

	waitKey(0);
	return 0;
}