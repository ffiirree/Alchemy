#ifndef ALCHEMY_IMGPROC_IMGPROC_H
#define ALCHEMY_IMGPROC_IMGPROC_H

#include <vector>
#include "core/defs.h"
#include "core/matrix.h"

#if defined(USE_OPENCV)
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif


typedef enum {
	DFT = -0x01, IDFT = 0x01
}Ft;

namespace alchemy{
template <typename _Tp> void cvtColor(const _Matrix<_Tp>&src, _Matrix<_Tp>&dst, int code);

// 多通道分离和混合
template <typename _Tp> void spilt(const _Matrix<_Tp> & src, std::vector<_Matrix<_Tp>> & mv);
template <typename _Tp> void merge(const _Matrix<_Tp> & src1, const _Matrix<_Tp> & src2, _Matrix<_Tp> & dst);
template <typename _Tp> void merge(const std::vector<_Matrix<_Tp>> & src, _Matrix<_Tp> & dst);

/**
 * @brief 上下颠倒图像
 * @param[in] src
 * @param[out] dst
 * @param[in] flags
 */
void convertImage(const Matrix8u *src, Matrix8u *dst, int flags = 0);

template <class _Tp> void copyMakeBorder(const _Matrix<_Tp> & src, _Matrix<_Tp> & dst, int top, int bottom, int left, int right);

/**
 * @berif fft
 * @param src Real
 * @param dst Complex
 */
void dft(const Matrix64f & src, Matrix64f & dst);

/**
 * @berif ifft
 * @param src Complex
 * @param dst Real
 */
void idft(Matrix64f & src, Matrix64f & dst);

///////////////////////////////////////////////Image Filtering/////////////////////////////////////////////////
///////////////////////////////////////////////Smoothing Images/////////////////////////////////////////////////
template <typename _T1, typename _T2> void conv(const _Matrix<_T1>& src, _Matrix<_T1>& dst, const _Matrix<_T2>& kernel, int borderType=BORDER_DEFAULT);

/**
 * @brief 均值滤波
 * \ kernel:
 * \                / 1 1 1 .. 1 \
 * \       1        | 1 1 1 .. 1 |
 * \    ------- =   | ....  .. 1 |
 * \    Kw * Kh     | ....  .. 1 |
 * \                \ 1 1 1 .. 1 /
 */
template <typename _Tp> void blur(_Matrix<_Tp>& src, _Matrix<_Tp>& dst, Size size, int borderType=BORDER_DEFAULT);
template <typename _Tp> void boxFilter(const _Matrix<_Tp>& src, _Matrix<_Tp>& dst, Size size, bool normalize = true, int borderType=BORDER_DEFAULT);

/**
 * @brief 获取高斯卷积核
 * \
 * \ if (sigmaX == 0) sigmaX = 0.3 * ((ksize.width - 1) * 0.5 - 1) + 0.8;
 * \ if (sigmaY == 0) sigmaY = 0.3 * ((ksize.height - 1) * 0.5 - 1) + 0.8;
 */
Matrix64f Gaussian(Size ksize, double sigmaX, double sigmaY);

/**
 * @brief 高斯滤波
 * @kernel -> Gassion()
 */
template <typename _Tp> void GaussianBlur(_Matrix<_Tp>&src, _Matrix<_Tp> & dst, Size size, double sigmaX = 0, double sigmaY = 0, int borderType = BORDER_DEFAULT);

template <typename _Tp> void embossingFilter(_Matrix<_Tp>& src, _Matrix<_Tp>&dst, Size size, int borderType = BORDER_DEFAULT);

template <typename _Tp> void medianFilter(_Matrix<_Tp>&src, _Matrix<_Tp>& dst, Size size, int borderType = BORDER_DEFAULT);

/**
 * @brief 双边滤波
 * @param[in] src
 * @param[out] dst
 * @param[in] size The kernel size, If it is non-positive, it is computed from sigmaSpace
 * @param[in] sigmaColor Filter sigma in the color space. A larger value of the parameter means that 
 *      farther colors within the pixel neighborhood (see sigmaSpace ) will be mixed together, 
 *      resulting in larger areas of semi-equal color.
 * @param[in] sigmaSpace Filter sigma in the coordinate space. A larger value of the parameter means 
 *      that farther pixels will influence each other as long as their colors are close enough (see sigmaColor ). 
 *      When d>0 , it specifies the neighborhood size regardless of sigmaSpace . 
 *      Otherwise, d is proportional to sigmaSpace .
 * @ret None
 */
template <typename _Tp> void bilateralFilter(const _Matrix<_Tp>&src, _Matrix<_Tp>&dst, int size, double sigmaColor, double sigmaSpace);

///////////////////////////////////////////////Morphology Transformations/////////////////////////////////////////
// 形态学滤波
template <typename _Tp> void morphOp(int code, _Matrix<_Tp>& src, _Matrix<_Tp>&dst, Size kernel, int borderType = BORDER_DEFAULT);
template <typename _Tp> void erode(_Matrix<_Tp>& src, _Matrix<_Tp>&dst, Size kernel, int borderType = BORDER_DEFAULT);
template <typename _Tp> void dilate(_Matrix<_Tp>& src, _Matrix<_Tp>&dst, Size kernel, int borderType = BORDER_DEFAULT);

// 形态学滤波的高级操作
template <typename _Tp> void morphEx(_Matrix<_Tp>& src, _Matrix<_Tp>&dst, int op, Size kernel, int borderType = BORDER_DEFAULT);
template <typename _Tp> void open(_Matrix<_Tp>& src, _Matrix<_Tp>&dst, Size kernel, int borderType = BORDER_DEFAULT);

///////////////////////////////////////////////Threshold/////////////////////////////////////////////////////////
/**
 * @breif 单通道固定阈值
 * @attention 单通道
 * @param[in] src
 * @param[out] dst
 * @param[in] thresh 阈值
 * @param[in] maxval
 * @param[in] type
 *          \ THRESH_BINARY         src(x, y) > thresh ? src(x, y) = maxval : src(x,y) = 0
 *          \ THRESH_BINARY_INV     src(x, y) > thresh ? src(x, y) = 0 : src(x,y) = maxval
 *          \ THRESH_TRUNC          src(x, y) > thresh ? src(x, y) = thresh : src(x,y) = src(x,y)
 *          \ THRESH_TOZERO         src(x, y) > thresh ? src(x, y) = src(x,y) : src(x,y) = 0
 *          \ THRESH_TOZERO_INV     src(x, y) > thresh ? src(x, y) = 0 : src(x, y) = src(x,y)
 * ret None
 */
template <typename _Tp> void threshold(_Matrix<_Tp> &src, _Matrix<_Tp>& dst, double thresh, double maxval, int type);

///////////////////////////////////////////////Image Pyramid/////////////////////////////////////////////////////
template <typename _Tp> void pyrUp(const _Matrix<_Tp>& src, _Matrix<_Tp>& dst);
template <typename _Tp> void pyrDown(const _Matrix<_Tp>& src, _Matrix<_Tp>& dst);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
enum  {
	GRAY_SCALE_INVARIANCE = 0x01,
	ROTATION_INVARIANCE = 0x02,
	UNIFORM_PATTERN = 0x04
};

// only: r = 1, P = 8
template <typename _Tp> void LBP(const _Matrix<_Tp>& src, _Matrix<_Tp>& dst, int r, int P, int mode = GRAY_SCALE_INVARIANCE, int borderType = BORDER_DEFAULT);
}

#include "imgproc.hpp"

#endif //! ALCHEMY_IMGPROC_IMGPROC_H