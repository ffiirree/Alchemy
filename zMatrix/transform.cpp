#include "transform.h"
#include <cmath>
namespace z {


void sobel(Matrix8u&src, Matrix8u&dst, int dx, int dy, int ksize)
{
	sobel(src, dst, Matrix8u(), dx, dy, ksize, true);
}

inline int only_max(Matrix8u&dstGD, int i, int j,  int k, int ksize, unsigned char ang)
{
	unsigned char * ptr = dstGD.ptr(i, j);
	unsigned char * ptr1;
	if (ang == 0) {
		for (int ii = 0; ii < ksize; ++ii) {
			ptr1 = dstGD.ptr(i, j - ksize / 2 + ii);
			if (ptr1 != 0) {
				if (ptr[k] < ptr1[k])
					return 0;
			}
		}
	}
	else if (ang == 45) {
		for (int ii = 0; ii < ksize; ++ii) {
			ptr1 = dstGD.ptr(i - ksize/2 + ii, j + ksize/2 - ii);
			if (ptr1 != 0) {
				if (ptr[k] < ptr1[k])
					return 0;
			}
		}
	}
	else if (ang == 90) {
		for (int ii = 0; ii < ksize; ++ii) {
			ptr1 = dstGD.ptr(i - ksize/2 + ii, j);
			if (ptr1 != 0) {
				if (ptr[k] < ptr1[k])
					return 0;
			}
		}
	}
	else if (ang == 135) {
		for (int ii = 0; ii < ksize; ++ii) {
			ptr1 = dstGD.ptr(i - ksize / 2 + ii, j - ksize / 2 + ii);
			if (ptr1 != 0) {
				if (ptr[k] < ptr1[k])
					return 0;
			}
		}
	}
	return 1;
}
/**
 * @berif sobel算子
 * @param[in] ksize, must be 1, 3, 5 or 7.
 */
void sobel(Matrix8u&src, Matrix8u&dst, Matrix8u&dstGD, int dx, int dy, int ksize, bool noGD)
{
	if (!src.equalSize(dst))
		dst.create(src.rows, src.cols, src.chs);
	if (!noGD)
		if (!dstGD.equalSize(src))
			dstGD.create(src.rows, src.cols, src.chs);

	Matrix8s Gx(ksize, ksize), Gy(ksize, ksize);

	switch (ksize) {
	case 1:
		
		break;

	case 3:
		Gx = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
		Gy = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
		break;

	case 5:
		break;

	case 7:
		break;

	default:
		_log_("Error ksize!")
		return;
	}
	

	int *tempGx = new int[src.chs];
	int *tempGy = new int[src.chs];
	int *tempG = new int[src.chs];
	int zerosx = 0, zerosy = 0;
	int m = ksize / 2, n = ksize / 2;
	unsigned char * srcPtr = nullptr;
	unsigned char * dstPtr = nullptr;
	unsigned char * dstGDPtr = nullptr;
	int alpha = 0;
	double ang = 0;

	for (int i = 0; i < dst.rows; ++i) {
		for (int j = 0; j < dst.cols; ++j) {

			memset(tempGx, 0, src.chs * sizeof(int));
			memset(tempGy, 0, src.chs * sizeof(int));
			memset(tempG, 0, src.chs * sizeof(int));
			zerosx = zerosy = 0;

			for (int ii = 0; ii < ksize; ++ii) {
				for (int jj = 0; jj < ksize; ++jj) {

					// 获取一个像素的地址
					srcPtr = src.ptr(i - m + ii, j - n + jj);

					if (srcPtr) {
						for (int k = 0; k < src.chs; ++k) {
							tempGx[k] += srcPtr[k] * Gx[ii][jj];
							tempGy[k] += srcPtr[k] * Gy[ii][jj];
						}
					}
					else {
						zerosx += Gx[ii][jj];
						zerosy += Gy[ii][jj];
					}

				} // !for(jj)
			} // !for(ii)

			if (zerosx != 0) {
				for (int k = 0; k < src.chs; ++k) {
					tempGx[k] /= zerosx;
					if (zerosy != 0) {
						tempGy[k] /= zerosy;
					}
				}
			}

			dstPtr = dst.ptr(i, j);
			if (!noGD)
				dstGDPtr = dstGD.ptr(i, j);

			for (int k = 0; k < src.chs; ++k) {
				dstPtr[k] = (unsigned char)std::sqrt(tempGx[k] * tempGx[k] + tempGy[k] * tempGy[k]);
				// 计算梯度方向
				if (!noGD) {
					ang = atan2(tempGy[k],tempGx[k]) * RAD2ANG;

					if ((ang > -22.5 && ang < 22.5) || (ang > 157.5 || ang < -157.5))
						dstGDPtr[k] = 0;
					else if ((ang > 22.5 && ang < 67.5) || (ang < -112.5 && ang > -157.5))
						dstGDPtr[k] = 45;
					else if ((ang > 67.5 && ang < 112.5) || (ang < -67.5 && ang > -112.5))
						dstGDPtr[k] = 90;
					else if ((ang < -22.5 && ang > -67.5) || (ang > 112.5 && ang  < 157.5))
						dstGDPtr[k] = 135;
				}
				if (!only_max(dstGD, i, j, k, ksize, dstGDPtr[k]))
					dstPtr[k] = 0;
				if (dstPtr[k] < 50)
					dstPtr[k] = 0;
			}


		} // !for(j)
	} // !for(i)

	delete[] tempGx;
	delete[] tempGy;
	delete[] tempG;
}
void double_threashold(Matrix8u&src, Matrix8u&dst, double threshold1, double threshold2)
{
	double maxt = threshold1 > threshold2 ? threshold1 : threshold2;
	double mint = threshold1 < threshold2 ? threshold1 : threshold2;

	if (!dst.equalSize(src))
		dst.create(src.rows, src.cols, src.chs);

	unsigned char *ptr, * ptr1, *ptr2, *ptr3, *ptr4, * dstPtr;

	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			ptr = src.ptr(i, j);
			ptr1 = src.ptr(i - 1, j);
			ptr2 = src.ptr(i, j - 1);
			ptr3 = src.ptr(i, j + 1);
			ptr4 = src.ptr(i + 1, j);
			dstPtr = dst.ptr(i, j);

			for (int k = 0; k < src.chs; ++k) {

				if (ptr[k] < mint) {
					dstPtr[k] = 0;
				}
				else if (ptr[k] > maxt) {
					dstPtr[k] = ptr[k];
				}
				else if ((ptr1 != 0 && ptr1[k] > maxt)
					|| (ptr2 != 0 && ptr2[k] > maxt)
					|| (ptr3 != 0 && ptr3[k] > maxt)
					|| (ptr4 != 0 && ptr4[k] > maxt)) {
					dstPtr[k] = ptr[k];
				}
				else {
					dstPtr[k] = 0;
				}
			}
		}
	}
}


void Canny(Matrix8u&src, Matrix8u&dst, double threshold1, double threshold2)
{
	Matrix8u dstGD;
	Matrix8u temp, temp1;
	GaussianBlur(src, temp, z::Size(5, 5));
	sobel(temp, temp1, dstGD);
	double_threashold(temp1, dst, threshold1, threshold2);
}




}

