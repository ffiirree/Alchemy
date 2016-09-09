#include "zimgproc.h"

namespace z{
Matrix8u Mat2Matrix8u(cv::Mat & mat)
{
	Matrix8u temp(mat.rows, mat.cols, mat.channels());
	memcpy(temp.data, mat.data, temp.size()*temp.chs);

	return temp;
}

}