#include <iostream>
#include <opencv2\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <string>
#include <ctime>  

#include "zcore\zcore.h"
#include "zimgproc\zimgproc.h"
#include "zimgproc\transform.h"
#include "zgui\zgui.h"
#include "zcore\debug.h"

using namespace std;
using namespace z;

int main(int argc, char *argv[])
{
	Matrix8u img = imread("test.jpeg");

	Rect r(1, 3, 4, 5);
	Rect a = r;
	Point3i p(2, 4, 6);

	line(img, Point(10, 20), Point(100, 68), Scalar(255, 255, 255));

	// œ‘ æ≤ …´Õº∆¨
	imshow("hello", img);
	waitKey(0);
	return 0;
}


