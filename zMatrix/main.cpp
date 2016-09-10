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
	Matrix8u mgray, mdis, mcolor = Mat2Matrix8u(src);

	timestamp.start();
	z::morphEx(mcolor, mdis, MORP_OPEN, z::Size(7, 7));
	timestamp.runtime();
	imshow("MORP_OPEN", Mat(mdis));

	timestamp.start();
	z::morphEx(mcolor, mdis, MORP_CLOSE, z::Size(7, 7));
	timestamp.runtime();
	imshow("MORP_CLOSE", Mat(mdis));


	timestamp.start();
	z::morphEx(mcolor, mdis, MORP_TOPHAT, z::Size(7, 7));
	timestamp.runtime();
	imshow("MORP_TOPHAT", Mat(mdis));

	timestamp.start();
	z::morphEx(mcolor, mdis, MORP_BLACKHAT, z::Size(7, 7));
	timestamp.runtime();
	imshow("MORP_BLACKHAT", Mat(mdis));

	

	timestamp.start();
	z::morphEx(mcolor, mdis, MORP_GRADIENT,z::Size(7, 7));
	timestamp.runtime();
	imshow("MORP_GRADIENT", Mat(mdis));

 	waitKey(0);
	return 0;
}