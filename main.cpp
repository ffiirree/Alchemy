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

int main(int argc, char *argv[])
{
	z::Matrix8u test = z::imread("test.jpeg");

	z::Matrix32s ker(3, 3, 1), ker2(3, 3, 1);
	z::Matrix8u dst, dst2;
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


	z::translation(test, ker, dst);
	z::translation(test, ker2, dst2);

	z::imshow("org", test);
	z::imshow("hou", dst);
	z::imshow("hou2", dst2);

	z::waitKey(0);

	return 0;
}