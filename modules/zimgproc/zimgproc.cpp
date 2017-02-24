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
#include <algorithm>
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
	memcpy(temp.data, mat.data, temp.size_*temp.chs);

	return temp;
}


void convertImage(Matrix8u *src, Matrix8u *dst, int flags)
{
	if (!dst->equalSize(*src))
		dst->create(src->rows, src->cols, src->chs);

	for (int i = 0; i < src->rows; ++i) 
		for (int j = 0; j < src->cols; ++j) 
			for (int k = 0; k < src->chs; ++k) 
				dst->ptr(i, j)[k] = src->ptr(src->rows - i - 1, j)[k];
}

void copyToArray(Matrix8u &src, char * arr)
{
	int dataSize = src.size_ * src.chs;
	for (int i = 0; i < dataSize; ++i) {
		arr[i] = src.data[i];
	}
}
/**
 * @berif 获取用于进行高斯滤波的高斯核
 */
Matrix64f Gassion(z::Size ksize, double sigmaX, double sigmaY)
{
    assert(ksize.width == ksize.height && ksize.width % 2 == 1);
    
    if (sigmaX == 0) sigmaX = 0.3 * ((ksize.width - 1) * 0.5 - 1) + 0.8;
	if (sigmaY == 0) sigmaY = 0.3 * ((ksize.height - 1) * 0.5 - 1) + 0.8;

	int x = ksize.width / 2;
	int y = ksize.height / 2;

    Matrix64f kernel(ksize);

    double alpha = 2 * Pi * sigmaX * sigmaY;

	for (int i = 0; i < kernel.rows; ++i) {
		for (int j = 0; j < kernel.cols; ++j) {
			auto z = std::pow((i - x), 2)/(2.0*std::pow(sigmaX, 2)) + std::pow((j - y), 2)/(2.0 * std::pow(sigmaY, 2));
			kernel[i][j] = exp(-z) / alpha;             // SUM(Ｇi,j) = 1
		}
	}
	return kernel;
}



/**
 * @berif 1D or 2D 离散傅里叶变换
 * @param src
 * @param dst
 */
void _dft(Matrix64f & src, Matrix64f & dst, Ft ft)
{
    Matrix64f temp(src.rows, src.cols, 2);
    Matrix64f end(src.rows, src.cols, 2);

	// 按层计算
	const int N = src.cols;
	for (int i = 0; i < src.rows; ++i) {
		for (int v = 0; v < N; ++v) {
			Complex mt(0, 0);
			for (int n = 0; n < N; ++n) {
				double beta = (2 * Pi * v * n) / N;
                Complex w(cos(beta), ft * sin(beta));
                Complex g(src.ptr(i, n)[0], src.ptr(i, n)[1]);
                mt += w * g;
			}
            if(ft == DFT){
                temp.ptr(i, v)[0] = mt.re;
                temp.ptr(i, v)[1] = mt.im;
            }
            else {
                temp.ptr(i, v)[0] = mt.re / N;
                temp.ptr(i, v)[1] = mt.im / N;
            }

		}
	}

    if (src.rows < 2) {
        dst = temp;
        return;
    }

	// 按列计算
	const int M = src.rows;
	for (int j = 0; j < src.cols; ++j) {
		for (int u = 0; u < M; ++u) {
			Complex mt(0,0);
			for (int m = 0; m < M; ++m) {
				double alpha = (2 * Pi * u * m) / M;

                Complex w(cos(alpha), ft * sin(alpha));
                Complex g(temp.ptr(m, j)[0], temp.ptr(m, j)[1]);
                mt += w * g;
			}
            if(ft == DFT){
                end.ptr(u, j)[0] = mt.re;
                end.ptr(u, j)[1] = mt.im;
            }
            else{
                end.ptr(u, j)[0] = mt.re / M;
                end.ptr(u, j)[1] = mt.im / M;
            }

		}
	}
    dst = end;
}


/**
 * @berif 1D或2D离散傅里叶变换
 */
void dft(Matrix64f & src, Matrix64f & dst)
{
    Matrix64f gx, gRe = src;
    Matrix64f gIm(src.rows, src.cols, 1);
	gIm.zeros();
	merge(gRe, gIm, gx);

	_dft(gx, dst, DFT);
}

void idft(Matrix64f & src, Matrix64f & dst)
{
    std::vector<Matrix64f> mv;
    Matrix64f temp;
    _dft(src, temp, IDFT);
    dst = temp;
}


/**
 * @berif 获取基2FFT理想的矩阵尺寸
 * @param cols
 * @return
 */
int getIdealCols(int cols)
{
    if(cols < 1){
        _log_("Error!");
    }

	int temp = 1;
	while (cols > temp) {
		temp *= 2;
	}
	return temp;
}
int getIdealRows(int rows)
{
	return getIdealCols(rows);
}

/**
 * @berif 对_Matrix类进行列的二进制反转
 * @param src
 */
void bitRevCols(Matrix64f & src)
{
	int32_t HELF_N = src.cols >> 1;
    int32_t k;

	for(int32_t i = 1, j = HELF_N; i < src.cols  - 1; ++i){
		if(i < j){
			for(int32_t row = 0; row < src.rows; ++row){
				src.swap(row, i, row, j);
			}
		}

        k = HELF_N;
        while(j >= k){
            j -= k;
            k >>= 1;
        }
        if(j < k) j += k;
	}
}

/**
 * @berif 对_Matrix的类对象进行行二进制反转
 * @param src
 */
void bitRevRows(Matrix64f & src)
{
    int32_t HELF_N = src.rows >> 1;
    int32_t k;

    for(int32_t i = 1, j = HELF_N; i < src.rows - 1; ++i){
        if(i < j){
            for(int32_t col = 0; col < src.cols; ++col){
                src.swap(i, col, j, col);
            }
        }

        k = HELF_N;
        while(j >= k){
            j -= k;
            k >>= 1;
        }
        if(j < k) j += k;
    }
}


/**
 * @berif 1D or 2D 基2FFT, 就地计算
 * @param src
 */
void _fft(Matrix64f & src, Ft ft)
{
    // 二进制反转，列反转
    bitRevCols(src);

	for (int i = 0; i < src.rows; ++i) {

        // 蝶形算法
		for (int l = 2; l <= src.cols; l <<= 1) {    // 需要log2(N)层
            for(int k = 0; k < src.cols; k += l) {
                for(int n = 0; n < (l >> 1); ++n) {

                    // Wn旋转因子
                    Complex W(cos((2 * Pi * n) / l), ft * sin((2 * Pi * n) / l));

                    // 上下蝶翅
                    Complex up(src.ptr(i, k + n)[0], src.ptr(i, k + n)[1]);
                    Complex down(src.ptr(i, k + n + l/2)[0], src.ptr(i, k + n + l/2)[1]);

                    Complex m = down * W;
                    down = up - m;
                    up = up + m;

                    src.ptr(i, k + n)[0] = up.re;
                    src.ptr(i, k + n)[1] = up.im;

                    src.ptr(i, k + n + l/2)[0] = down.re;
                    src.ptr(i, k + n + l/2)[1] = down.im;

                } // !for(n)
            } // !for(k)
		} // !for(l)
	} // !for(i)

    // 如果是1D的矩阵，则返回
    if(src.rows < 2) return;

    // 行反转
    bitRevRows(src);

    for(int j = 0; j < src.cols; ++j){

        for (int l = 2; l <= src.rows; l <<= 1) {    // 需要log2(N)层
            for (int k = 0; k < src.rows; k += l) {
                for (int n = 0; n < (l >> 1); ++n) {

                    // W = cos(2 * Pi / N) - sin(2 * Pi / N)
                    Complex W(cos((2 * Pi * n) / l), ft * sin((2 * Pi * n) / l));

                    Complex up(src.ptr(k + n, j)[0], src.ptr(k + n, j)[1]);
                    Complex down(src.ptr(k + n + l/2, j)[0], src.ptr(k + n + l/2, j)[1]);

                    Complex m = down * W;
                    down = up - m;
                    up = up + m;

                    src.ptr(k + n, j)[0] = up.re;
                    src.ptr(k + n, j)[1] = up.im;

                    src.ptr(k + n + l/2, j)[0] = down.re;
                    src.ptr(k + n + l/2, j)[1] = down.im;
                }
            }
        }
    }
}



void fft(Matrix64f & src, Matrix64f & dst)
{
    Matrix64f gRe;
	int fft_rows = getIdealRows(src.rows);
	int fft_cols = getIdealCols(src.cols);

    // 扩充原图像
	copyMakeBorder(src, gRe, 0, fft_rows - src.rows, 0, fft_cols - src.cols);

    // 虚数部分
    Matrix64f gIm(fft_rows, fft_cols, 1);
    gIm.zeros();

    // 虚数和实数部分合成，FFT输入的图像
    Matrix64f gx;
	merge(gRe, gIm, gx);

    // 进行快速傅里叶变换
	_fft(gx, DFT);
    dst = gx;
}

void ifft(Matrix64f & src, Matrix64f & dst)
{
    std::vector<Matrix64f> mv;
    _fft(src, IDFT);
    
    for (size_t i = 0; i < src.size_ * 2; ++i) {
        *(src.data + i) /= src.size_;
    }
    dst = src;
}


}