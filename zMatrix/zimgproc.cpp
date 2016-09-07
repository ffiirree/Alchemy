#include "zimgproc.h"

Matrix8u Mat2Matrix8u(cv::Mat & mat)
{
	Matrix8u temp(mat.rows, mat.cols);
	memcpy(temp.data, mat.data, temp.size());

	return temp;
}
