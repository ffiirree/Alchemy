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

#ifdef USE_FFTW
#include <fftw3.h>
#endif

namespace z {

void convertImage(Matrix8u *src, Matrix8u *dst, int flags)
{
    __unused_parameter__(flags);

    if (dst->shape() != src->shape())
        dst->create(src->shape());

    for (auto i = 0; i < src->rows; ++i)
        for (auto j = 0; j < src->cols; ++j)
            for (auto k = 0; k < src->channels(); ++k)
                dst->at(i, j, k) = src->at(src->rows - i - 1, j, k);
}

Matrix64f Gaussian(Size ksize, double sigmaX, double sigmaY)
{
    assert(ksize.width == ksize.height && ksize.width % 2 == 1);

    if (sigmaX == 0) sigmaX = 0.3 * ((ksize.width - 1) * 0.5 - 1) + 0.8;
    if (sigmaY == 0) sigmaY = 0.3 * ((ksize.height - 1) * 0.5 - 1) + 0.8;

    int x = ksize.width / 2;
    int y = ksize.height / 2;

    Matrix64f kernel(ksize);

    double alpha = 0;

    for (int i = 0; i < kernel.rows; ++i) {
        for (int j = 0; j < kernel.cols; ++j) {
            auto z = std::pow((i - x), 2)/(2.0 * std::pow(sigmaX, 2)) + std::pow((j - y), 2)/(2.0 * std::pow(sigmaY, 2));
            alpha += kernel.at<double>(i, j) = exp(-z);
        }
    }

    // SUM(Gi,j) = 1
    for(auto& i : kernel) {
        i /= alpha;
    }
    return kernel;
}

///////////////////////////////////////////// DFT ///////////////////////////////////////////////////
#ifndef USE_FFTW
void bitRevCols(Matrix64f & src);
void bitRevRows(Matrix64f & src);

static int getIdealRows(int rows);
static int getIdealCols(int cols);
static void _dft(Matrix64f & src, Matrix64f & dst, Ft ft);
/**
 * @brief 1D or 2D 离散傅里叶变换
 * @param src
 * @param dst
 */
void _dft(Matrix64f & src, Matrix64f & dst, Ft ft)
{
    Matrix64f temp(src.rows, src.cols, 2);
    Matrix64f end(src.rows, src.cols, 2);

    // 按层计算
    const auto N = src.cols;
    for (int i = 0; i < src.rows; ++i) {
        for (int v = 0; v < N; ++v) {
            Complex mt(0, 0);
            for (int n = 0; n < N; ++n) {
                double beta = (2 * Pi * v * n) / N;
                Complex w(cos(beta), ft * sin(beta));
                Complex g(src.ptr(i, n)[0], src.ptr(i, n)[1]);
                mt += w * g;
            }
            if (ft == DFT) {
                temp.at<Complex>(i, v) = mt;
            }
            else {
                temp.at<Complex>(i, v) = mt / static_cast<double>(N);
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
            Complex mt(0, 0);
            for (int m = 0; m < M; ++m) {
                double alpha = (2 * Pi * u * m) / M;

                Complex w(cos(alpha), ft * sin(alpha));
                Complex g(temp.ptr(m, j)[0], temp.ptr(m, j)[1]);
                mt += w * g;
            }
            if (ft == DFT) {
                end.at<Complex>(u, j) = mt;
            }
            else {
                end.at<Complex>(u, j) = mt / static_cast<double>(M);
            }

        }
    }
    dst = end;
}

/**
 * @brief 获取基2FFT理想的矩阵尺寸
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
 * @brief 对_Matrix类进行列的二进制反转
 * @param src
 */
void bitRevCols(Matrix64f & src)
{
	auto HELF_N = src.cols >> 1;

	for (auto i = 1, j = HELF_N; i < src.cols - 1; ++i) {
        if (i < j) {
            for (auto row = 0; row < src.rows; ++row) {
                src.swap(row, i, row, j);
            }
        }

		auto k = HELF_N;
        while (j >= k) {
            j -= k;
            k >>= 1;
        }
        if (j < k) j += k;
    }
}

/**
 * @brief 对_Matrix的类对象进行行二进制反转
 * @param src
 */
void bitRevRows(Matrix64f & src)
{
	auto HELF_N = src.rows >> 1;

	for (auto i = 1, j = HELF_N; i < src.rows - 1; ++i) {
        if (i < j) {
            for (auto col = 0; col < src.cols; ++col) {
                src.swap(i, col, j, col);
            }
        }

		auto k = HELF_N;
        while (j >= k) {
            j -= k;
            k >>= 1;
        }
        if (j < k) j += k;
    }
}


/**
 * @brief 1D or 2D 基2FFT, 就地计算
 * @param src
 */
void _fft(Matrix64f & src, Ft ft)
{
    // 二进制反转，列反转
    bitRevCols(src);

    for (int i = 0; i < src.rows; ++i) {

        // 蝶形算法
        for (int l = 2; l <= src.cols; l <<= 1) {    // 需要log2(N)层
            for (int k = 0; k < src.cols; k += l) {
                for (int n = 0; n < (l >> 1); ++n) {

                    // Wn旋转因子
                    Complex W(cos((2 * Pi * n) / l), ft * sin((2 * Pi * n) / l));

                    // 上下蝶翅
                    Complex up(src.ptr(i, k + n)[0], src.ptr(i, k + n)[1]);
                    Complex down(src.ptr(i, k + n + l / 2)[0], src.ptr(i, k + n + l / 2)[1]);

	                auto m = down * W;
                    down = up - m;
                    up = up + m;

                    src.at<Complex>(i, k + n) = up;
                    src.at<Complex>(i, k + n + l / 2) = down;

                } // !for(n)
            } // !for(k)
        } // !for(l)
    } // !for(i)

      // 如果是1D的矩阵，则返回
    if (src.rows < 2) return;

    // 行反转
    bitRevRows(src);

    for (int j = 0; j < src.cols; ++j) {

        for (int l = 2; l <= src.rows; l <<= 1) {    // 需要log2(N)层
            for (int k = 0; k < src.rows; k += l) {
                for (int n = 0; n < (l >> 1); ++n) {

                    // W = cos(2 * Pi / N) - sin(2 * Pi / N)
                    Complex W(cos((2 * Pi * n) / l), ft * sin((2 * Pi * n) / l));

                    Complex up(src.ptr(k + n, j)[0], src.ptr(k + n, j)[1]);
                    Complex down(src.ptr(k + n + l / 2, j)[0], src.ptr(k + n + l / 2, j)[1]);

                    auto m = down * W;
                    down = up - m;
                    up = up + m;

                    src.at<Complex>(k + n, j) = up;
                    src.at<Complex>(k + n + l / 2, j) = down;
                }
            }
        }
    }
}
#endif //! USE_FFTW

void dft(const Matrix64f & src, Matrix64f & dst)
{
#ifdef USE_FFTW
    dst = Matrix64f::zeros(src.rows, src.cols, 2);

    const auto plan = fftw_plan_dft_r2c_2d(src.rows, src.cols, const_cast<double *>(src.ptr()), reinterpret_cast<fftw_complex *>(dst.ptr()), FFTW_ESTIMATE);
    fftw_execute(plan);

    // Destory and cleanup.
    fftw_destroy_plan(plan);
    fftw_cleanup();
#else
    Matrix64f gRe;
    int fft_rows = getIdealRows(src.rows);
    int fft_cols = getIdealCols(src.cols);

    // 扩充原图像
    copyMakeBorder(src, gRe, 0, fft_rows - src.rows, 0, fft_cols - src.cols);

    // 虚数部分
    Matrix64f gIm(fft_rows, fft_cols, 1);
    gIm = 0;

    // 虚数和实数部分合成，FFT输入的图像
    merge(gRe, gIm, dst);

    // 进行快速傅里叶变换
    _fft(dst, DFT);
#endif
}


void idft(Matrix64f & src, Matrix64f & dst)
{
#ifdef USE_FFTW
    dst = Matrix64f::zeros(src.rows, src.cols, 1);

    const auto p3 = fftw_plan_dft_c2r_2d(src.rows, src.cols, reinterpret_cast<fftw_complex *>(src.ptr()), reinterpret_cast<double *>(dst.ptr()), FFTW_ESTIMATE);
    fftw_execute(p3);
    dst /= dst.total();

    fftw_destroy_plan(p3);
    fftw_cleanup();
#else
    src.copyTo(dst);
    _fft(dst, IDFT);

    for(auto& pixel : _Matrix<Complex>(dst)) {
        pixel /= dst.total();
    }
#endif
}


}
