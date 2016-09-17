/**
 ******************************************************************************
 * @file    zimgproc.cpp
 * @author  zlq
 * @version V1.0
 * @date    2016.9.14
 * @brief   与类型无关的图像处理函数的实现
 ******************************************************************************
 * @attention
 *
 *
 ******************************************************************************
 */
#include "zimgproc.h"
#include "debug.h"

namespace z{

/**
 * @berif openCV中的Mat类转换为Matrix8u类
 */
Matrix8u Mat2Matrix8u(cv::Mat & mat)
{
	Matrix8u temp(mat.rows, mat.cols, mat.channels());
	memcpy(temp.data, mat.data, temp.size()*temp.chs);

	return temp;
}

/**
 * @berif 获取用于进行高斯滤波的高斯核
 */
Matrix Gassion(z::Size ksize, double sigmaX, double sigmaY)
{
	if (ksize.width != ksize.height || ksize.width % 2 != 1) {
		_log_("ksize.width != ksize.height || ksize.width % 2 != 1");
	}

	if (sigmaX == 0) sigmaX = ksize.width / 2.0;
	if (sigmaY == 0) sigmaY = ksize.height / 2.0;

	int x = ksize.width / 2;
	int y = ksize.height / 2;
	double z;

	Matrix kernel(ksize);

	for (int i = 0; i < kernel.rows; ++i) {
		for (int j = 0; j < kernel.cols; ++j) {
			z = (i - x) * (i - x)/sigmaX + (j - y) * (j - y)/sigmaY;
			kernel[i][j] = exp(-z);
		}
	}

	double a = 1.0 / kernel[0][0];

	for (int i = 0; i < kernel.rows; ++i) {
		for (int j = 0; j < kernel.cols; ++j) {
			kernel[i][j] = int(kernel[i][j] * a);
		}
	}
	return kernel;
}
}