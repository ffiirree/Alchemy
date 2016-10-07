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
#include "zcore\zcore.h"

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
template <class _type> void cvtColor(_Matrix<_type>&src, _Matrix<_type>&dst, int code);

// 多通道分离和混合
template <class _type> void spilt(_Matrix<_type> & src, std::vector<_Matrix<_type>> & mv);
template <class _type> void merge(_Matrix<_type> & src1, _Matrix<_type> & src2, _Matrix<_type> & dst);
template <class _type> void merge(std::vector<_Matrix<_type>> & src, _Matrix<_type> & dst);

void convertImage(Matrix8u *src, Matrix8u *dst, int flags = 0);
void copyToArray(Matrix8u &src, char * arr);

template <class _type> void copyMakeBorder(_Matrix<_type> & src, _Matrix<_type> & dst, int top, int bottom, int left, int right);

// 离散傅里叶DFT
void _dft(Matrix & src);
void dft(Matrix & src, Matrix & dst);
void idft(Matrix & src, Matrix & dst);

void bitRevCols(Matrix & src);
void bitRevRows(Matrix & src);



// 快速傅里叶变换FFT
void _fft(Matrix & src, Ft ft);
void fft(Matrix & src, Matrix & dst);
void ifft(Matrix & src, Matrix & dst);

// 线性滤波
template <class _type> void blur(_Matrix<_type>& src, _Matrix<_type>& dst, Size size);
template <class _type> void boxFilter(const _Matrix<_type>& src, _Matrix<_type>& dst, Size size, bool normalize);
template <class _type> void GaussianBlur(_Matrix<_type>&src, _Matrix<_type> & dst, Size size, double sigmaX = 0, double sigmaY = 0);
template <class _type> _Matrix<_type> embossingFilter(_Matrix<_type> src, Size size, float ang);
template <class _type> _Matrix<_type> edgeDetection(_Matrix<_type> src, Size size, float ang);
template <class _type> _Matrix<_type> motionBlur(_Matrix<_type> src, Size size, float ang);

// 非线性滤波
template <class _type> void medianFilter(_Matrix<_type>&src, _Matrix<_type>& dst, Size size);
Matrix Gassion(z::Size ksize, double sigmaX, double sigmaY);

// 形态学滤波
template <class _type> void morphOp(int code, _Matrix<_type>& src, _Matrix<_type>&dst, Size kernel);
template <class _type> void erode(_Matrix<_type>& src, _Matrix<_type>&dst, Size kernel);
template <class _type> void dilate(_Matrix<_type>& src, _Matrix<_type>&dst, Size kernel);

//形态学滤波的高级操作
template <class _type> void morphEx(_Matrix<_type>& src, _Matrix<_type>&dst, int op, Size kernel);
template <class _type> void open(_Matrix<_type>& src, _Matrix<_type>&dst, Size kernel);
}

#endif // !__cplusplus

#include "zimgproc.hpp"

#endif