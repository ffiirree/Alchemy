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
#include "zcore.h"
#include "zmatch\zmatch.h"

#if defined(OPENCV)
#include <opencv2\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#endif

#ifdef __cplusplus
namespace z{

template<class _Tp> class _Size;

template<class _Tp> class _Size
{
public:
	typedef _Tp value_type;

	//! various constructors
	_Size();
	_Size(_Tp _width, _Tp _height);
	_Size(const _Size& sz);

	_Size& operator = (const _Size& sz);
	//! the area (width*height)
	_Tp area() const;

	//! conversion of another data type.
	template<class _Tp2> operator _Size<_Tp2>() const;

	_Tp width, height; // the width and the height
};
template<class _Tp> inline _Size<_Tp>::_Size() :width(0), height(0) {}
template<class _Tp> inline _Size<_Tp>::_Size(_Tp _width, _Tp _height) : width(_width), height(_height) {}
template<class _Tp> inline _Size<_Tp>::_Size(const _Size& sz) : width(sz.width), height(sz.height) {}
template<class _Tp> inline _Tp _Size<_Tp>::area() const { return width * height; }


typedef _Size<int>      Size2i;
typedef _Size<double>   Size2d;
typedef _Size<float>    Sizef;
typedef _Size<int>      Size;

Matrix8u Mat2Matrix8u(cv::Mat & mat);
template <class _type> void cvtColor(_Matrix<_type>&src, _Matrix<_type>&dst, int code);

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