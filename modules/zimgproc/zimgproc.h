/**
 ******************************************************************************
 * @file    zimgproc.h
 * @author  zlq
 * @version V1.0
 * @date    2016.9.14
 * @brief   图像处理的函数定义
 ******************************************************************************
 * @attention
 *
 *
 ******************************************************************************
 */
#ifndef _ZIMGPROC_H
#define _ZIMGPROC_H

#include <string>
#include <vector>
#include "zcore\zmatrix.h"

#if defined(OPENCV)
#include <opencv2\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#endif


typedef enum {
	DFT = -0x01, IDFT = 0x01
}Ft;

#ifdef __cplusplus
namespace z{
Matrix8u Mat2Matrix8u(cv::Mat & mat);

template <typename _Tp> void cvtColor(const _Matrix<_Tp>&src, _Matrix<_Tp>&dst, int code);

// 多通道分离和混合
template <typename _Tp> void spilt(_Matrix<_Tp> & src, std::vector<_Matrix<_Tp>> & mv);
template <typename _Tp> void merge(_Matrix<_Tp> & src1, _Matrix<_Tp> & src2, _Matrix<_Tp> & dst);
template <typename _Tp> void merge(std::vector<_Matrix<_Tp>> & src, _Matrix<_Tp> & dst);

/**
 * @berif 上下颠倒图像
 * @param[in] src
 * @param[out] dst
 * @param[in] flags
 */
void convertImage(Matrix8u *src, Matrix8u *dst, int flags = 0);
void copyToArray(Matrix8u &src, char * arr);

template <class _Tp> void copyMakeBorder(_Matrix<_Tp> & src, _Matrix<_Tp> & dst, int top, int bottom, int left, int right);

// 离散傅里叶DFT
void dft(Matrix64f & src, Matrix64f & dst);
void idft(Matrix64f & src, Matrix64f & dst);

void bitRevCols(Matrix64f & src);
void bitRevRows(Matrix64f & src);



// 快速傅里叶变换FFT
void _fft(Matrix64f & src, Ft ft);
void fft(Matrix64f & src, Matrix64f & dst);
void ifft(Matrix64f & src, Matrix64f & dst);

///////////////////////////////////////////////Image Filtering/////////////////////////////////////////////////
///////////////////////////////////////////////Smoothing Images/////////////////////////////////////////////////
/**
 * @berif 均值滤波
 * \ kernel:
 * \                / 1 1 1 .. 1 \
 * \       1        | 1 1 1 .. 1 |
 * \    ------- =   | ....  .. 1 |
 * \    Kw * Kh     | ....  .. 1 |
 * \                \ 1 1 1 .. 1 /
 */
template <typename _Tp> void blur(_Matrix<_Tp>& src, _Matrix<_Tp>& dst, Size size);
template <typename _Tp> void boxFilter(const _Matrix<_Tp>& src, _Matrix<_Tp>& dst, Size size, bool normalize = true);

/**
 * @berif 获取高斯卷积核
 * \
 * \ if (sigmaX == 0) sigmaX = 0.3 * ((ksize.width - 1) * 0.5 - 1) + 0.8;
 * \ if (sigmaY == 0) sigmaY = 0.3 * ((ksize.height - 1) * 0.5 - 1) + 0.8;
 */
Matrix64f Gassion(z::Size ksize, double sigmaX, double sigmaY);

/**
 * @berif 高斯滤波
 * @kernel -> Gassion()
 */
template <typename _Tp> void GaussianBlur(_Matrix<_Tp>&src, _Matrix<_Tp> & dst, Size size, double sigmaX = 0, double sigmaY = 0);

template <typename _Tp> void embossingFilter(_Matrix<_Tp>& src, _Matrix<_Tp>&dst, Size size);

template <typename _Tp> void medianFilter(_Matrix<_Tp>&src, _Matrix<_Tp>& dst, Size size);

/**
 * @berif 双边滤波
 * @param[in] src
 * @param[out] dst
 * @param[in] size, the kernel size, If it is non-positive, it is computed from sigmaSpace
 * @param[in] sigmaColor, Filter sigma in the color space. A larger value of the parameter means that 
 *      farther colors within the pixel neighborhood (see sigmaSpace ) will be mixed together, 
 *      resulting in larger areas of semi-equal color.
 * @param[in] sigmaSpace, Filter sigma in the coordinate space. A larger value of the parameter means 
 *      that farther pixels will influence each other as long as their colors are close enough (see sigmaColor ). 
 *      When d>0 , it specifies the neighborhood size regardless of sigmaSpace . 
 *      Otherwise, d is proportional to sigmaSpace .
 * @ret None
 */
template <typename _Tp> void bilateralFilter(const _Matrix<_Tp>&src, _Matrix<_Tp>&dst, int size, double sigmaColor, double sigmaSpace);

///////////////////////////////////////////////Image Sharpening/////////////////////////////////////////
template <typename _Tp> void Laplacian(const _Matrix<_Tp>&src, _Matrix<_Tp>&dst, int ksize = 1);


///////////////////////////////////////////////Morphology Transformations/////////////////////////////////////////
// 形态学滤波
template <typename _Tp> void morphOp(int code, _Matrix<_Tp>& src, _Matrix<_Tp>&dst, Size kernel);
template <typename _Tp> void erode(_Matrix<_Tp>& src, _Matrix<_Tp>&dst, Size kernel);
template <typename _Tp> void dilate(_Matrix<_Tp>& src, _Matrix<_Tp>&dst, Size kernel);

// 形态学滤波的高级操作
template <typename _Tp> void morphEx(_Matrix<_Tp>& src, _Matrix<_Tp>&dst, int op, Size kernel);
template <typename _Tp> void open(_Matrix<_Tp>& src, _Matrix<_Tp>&dst, Size kernel);

///////////////////////////////////////////////Threshold/////////////////////////////////////////////////////////
/**
 * @breif 单通道固定阈值
 * @attention 单通道
 * @param[in] src
 * @param[out] dst
 * @param[in] thresh, 阈值
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
template <typename _Tp> void pyrUp(_Matrix<_Tp>& src, _Matrix<_Tp>& dst);
template <typename _Tp> void pyrDown(_Matrix<_Tp>& src, _Matrix<_Tp>& dst);
}

#endif // !__cplusplus

#include "zimgproc.hpp"

#endif