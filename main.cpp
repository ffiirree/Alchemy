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
	Matrix8u test = imread("test.jpeg");

	//line(test, Point(0, 0), Point(300, 400), Scalar(255, 0, 0));

	//imshow("test", test);
	//waitKey(0);

	Matrix32s ker(3, 3, 1), ker2(3, 3, 1);
	Matrix8u dst, dst2;
	ker = {
		1, 0, 0,
		0, 1, 0,
		-200, -200, 1
	};
	ker2 = {
		1, 0, 0,
		0, 1, 0,
		200, 200, 1
	};


	remap(test, ker, dst);
	remap(test, ker2, dst2);

	imshow("org", test);
	imshow("hou", dst);
	imshow("hou2", dst2);

	waitKey(0);

	return 0;
}