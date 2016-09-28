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
#include "zcore\debug.h"
#include "zmatch\zmatch.h"

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
* @berif 上下颠倒图像
*/
void convertImage(Matrix8u *src, Matrix8u *dst, int flags)
{
	if (!dst->equalSize(*src))
		dst->create(src->rows, src->cols, src->chs);

	int rows = src->rows;
	int cols = src->cols;

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			for (int k = 0; k < src->chs; ++k) {
				dst->ptr(i, j)[k] = src->ptr(rows - i - 1, j)[k];
			}
		}
	}
}

void copyToArray(Matrix8u &src, char * arr)
{
	int dataSize = src.size()* src.chs;
	for (size_t i = 0; i < dataSize; ++i) {
		arr[i] = src.data[i];
	}
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



void _dft(Matrix & src, Matrix & dst)
{
	Matrix temp(src.rows, src.cols, 2);
	Matrix end(src.rows, src.cols, 2);

	// 按层计算
	const int N = src.cols;
	for (int i = 0; i < src.rows; ++i) {
		for (int v = 0; v < N; ++v) {
			double Re = 0;
			double Im = 0;
			for (int n = 0; n < N; ++n) {
				double beta = (2 * Pi * v * n) / N;
				double sinx = sin(beta);
				double cosx = cos(beta);

				double gRe = src.ptr(i, n)[0];
				double gIm = src.ptr(i, n)[1];

				Re = Re + gRe * cosx + gIm * sinx;
				Im = Im - (gRe * sinx + gIm * cosx);
			}
			temp.ptr(i, v)[0] = Re;
			temp.ptr(i, v)[1] = Im;
		}
	}
	
	// 按列计算
	const int M = src.rows;
	for (int j = 0; j < src.cols; ++j) {
		for (int u = 0; u < M; ++u) {
			double Re = 0;
			double Im = 0;
			for (int m = 0; m < M; ++m) {
				double alpha = (2 * Pi * u * m) / M;
				double sinx = sin(alpha);
				double cosx = cos(alpha);

				double gRe = temp.ptr(m, j)[0];
				double gIm = temp.ptr(m, j)[1];

				// 复数乘法
				// (gRe - gIm * i)·(cosx - sinx * i) = (gRe * cosx + gIm * sinx) - (gRe * sinx + gIm * cosx) * i
				Re = Re + gRe * cosx + gIm * sinx;
				Im = Im + (gRe * sinx + gIm * cosx);
			}
			end.ptr(u, j)[0] = Re;
			end.ptr(u, j)[1] = Im;
		}
	}

	dst = end;
}

/**
 * @berif 1D或2D离散傅里叶变换
 */
void dft(Matrix8u & src, Matrix & dst)
{
	Matrix gRe = src;
	Matrix gIm(src.rows, src.cols, 1);
	Matrix gx;
	merge(gRe, gIm, gx);


	_dft(gx, dst);
}


int getIdealCols(int cols)
{
	int temp = 1;
	while (cols >= temp) {
		temp *= 2;
	}
	return temp;
}
int getIdealRows(int rows)
{
	return getIdealCols(rows);
}


//void _fft(Matrix & src, Matrix & dst)
//{
//	Matrix temp(src.rows, src.cols, 2);
//	Matrix end(src.rows, src.cols, 2);
//
//
//	// 按层FFT
//	const int L = log(src.cols)/log(2);                    // 需要log2(N)层
//	const int N = src.cols;
//	for (int i = 0; i < src.rows; ++i) {
//		for (int l = 0; l < L; ++l) {
//
//			for (int n = 0; n < N; ++n) {
//
//				double wRe = cos((2 * Pi * n) / N);
//				double wIm = sin((2 * Pi * n) / N);
//
//			}
//
//
//		}
//	}
//}



//void fft(Matrix8u & src, Matrix & dst)
//{
//	Matrix gRe;
//	int fft_rows = getIdealRows(src.rows);
//	int fft_cols = getIdealCols(src.cols);
//	copyMakeBorder(Matrix(src), gRe, 0, fft_rows - src.rows, 0, fft_cols - src.cols);
//	
//	Matrix gIm(fft_rows, fft_cols, 1);
//
//	// 
//	Matrix gx;
//	merge(gRe, gIm, gx);
//
//	_fft(gx, dst);
//}

}