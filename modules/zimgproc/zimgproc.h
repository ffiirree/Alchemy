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




//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Matrix8u Mat2Matrix8u(cv::Mat & mat);
template <class _Tp> void cvtColor(const _Matrix<_Tp>&src, _Matrix<_Tp>&dst, int code);

// 多通道分离和混合
template <class _Tp> void spilt(_Matrix<_Tp> & src, std::vector<_Matrix<_Tp>> & mv);
template <class _Tp> void merge(_Matrix<_Tp> & src1, _Matrix<_Tp> & src2, _Matrix<_Tp> & dst);
template <class _Tp> void merge(std::vector<_Matrix<_Tp>> & src, _Matrix<_Tp> & dst);

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
void _dft(Matrix64f & src);
void dft(Matrix64f & src, Matrix64f & dst);
void idft(Matrix64f & src, Matrix64f & dst);

void bitRevCols(Matrix64f & src);
void bitRevRows(Matrix64f & src);



// 快速傅里叶变换FFT
void _fft(Matrix64f & src, Ft ft);
void fft(Matrix64f & src, Matrix64f & dst);
void ifft(Matrix64f & src, Matrix64f & dst);

// 线性滤波
template <class _Tp> void blur(_Matrix<_Tp>& src, _Matrix<_Tp>& dst, Size size);
template <class _Tp> void boxFilter(const _Matrix<_Tp>& src, _Matrix<_Tp>& dst, Size size, bool normalize);
template <class _Tp> void GaussianBlur(_Matrix<_Tp>&src, _Matrix<_Tp> & dst, Size size, double sigmaX = 0, double sigmaY = 0);
template <class _Tp> _Matrix<_Tp> embossingFilter(_Matrix<_Tp> src, Size size, float ang);
template <class _Tp> _Matrix<_Tp> edgeDetection(_Matrix<_Tp> src, Size size, float ang);
template <class _Tp> _Matrix<_Tp> motionBlur(_Matrix<_Tp> src, Size size, float ang);

// 非线性滤波
template <class _Tp> void medianFilter(_Matrix<_Tp>&src, _Matrix<_Tp>& dst, Size size);
Matrix64f Gassion(z::Size ksize, double sigmaX, double sigmaY);

// 形态学滤波
template <class _Tp> void morphOp(int code, _Matrix<_Tp>& src, _Matrix<_Tp>&dst, Size kernel);
template <class _Tp> void erode(_Matrix<_Tp>& src, _Matrix<_Tp>&dst, Size kernel);
template <class _Tp> void dilate(_Matrix<_Tp>& src, _Matrix<_Tp>&dst, Size kernel);

//形态学滤波的高级操作
template <class _Tp> void morphEx(_Matrix<_Tp>& src, _Matrix<_Tp>&dst, int op, Size kernel);
template <class _Tp> void open(_Matrix<_Tp>& src, _Matrix<_Tp>&dst, Size kernel);


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
}

#endif // !__cplusplus

#include "zimgproc.hpp"

#endif