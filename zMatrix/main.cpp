#include <iostream>

#include"config_default.h"
#include "zmatrix.h"
#include "zimgproc.h"
#include <opencv2\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
	Mat gray, display, src = imread("test.jpeg");
	cvtColor(src, gray, CV_BGR2GRAY);


	imshow("src", src);
	imshow("gray", gray);

	Matrix8u m = Mat2Matrix8u(gray);

	// 卷积核
	Matrix64f core(3, 3);
	// 均值滤波
	core = {
		 0.0, 0.2, 0.0,
		 0.2, 0.0, 0.2,
		 0.0, 0.2, 0.0
	};
	m = m.conv(core);               // 或者:display = conv(m, core);

	imshow("filter", display);

	// 测试边缘检测
	core = {
		-0.1, -0.1, -0.1,
		-0.1,  0.8, -0.1,
		-0.1, -0.1, -0.1
	};
	display = conv(m, core);		// 或者:display = m.conv(core);

	imshow("edge", display);

	waitKey(0);
	return 0;
}