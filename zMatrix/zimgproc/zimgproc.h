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
#include "zcore.h"

#if defined(OPENCV)
#include <opencv2\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#endif

#ifdef __cplusplus
namespace z{

template<class _Tp> class _Rect;

////////////////////////////////////////_Complex/////////////////////////////////////////////
template<class _Tp> class _Complex
{
public:
	_Complex();
	_Complex( _Tp _re, _Tp _im=0 );
	_Complex( const std::complex<_Tp>& c );

	_Complex<_Tp> conj() const;                        // 共轭

    _Tp re, im;
};
template<class _Tp> inline _Complex<_Tp>::_Complex() :re(0), im(0) {}
template<class _Tp> inline _Complex<_Tp>::_Complex(_Tp _re, _Tp _im = 0) : re(_re), im(_im) { }
template<class _Tp> inline _Complex<_Tp>::_Complex(const std::complex<_Tp>& c) : re(c.re), im(c.im) { }
template<class _Tp> inline _Complex<_Tp> _Complex<_Tp>::conj() const { return _Complex<_tp>(re, -im); }


typedef _Complex<float> Complexf;
typedef _Complex<double> Complexd;

/////////////////////////////////////////_Point////////////////////////////////////////////
template<class _Tp> class _Point
{
public:
	_Point();
	_Point(_Tp _x, _Tp _y);
	_Point(const _Point& pt);

	_Point& operator = (const _Point& pt);
    
    _Tp dot(const _Point& pt) const;                    // 点乘
    double cross(const _Point& pt) const;               // 叉积
    bool inside(const _Rect<_Tp>& r) const;             // 检查点是否在区域内

    _Tp x, y;
};
template<class _Tp> inline _Point<_Tp>::_Point() : x(0), y(0) { }
template<class _Tp> inline _Point<_Tp>::_Point(_Tp _x, _Tp _y) : x(_x), y(_y) { }
template<class _Tp> inline _Point<_Tp>::_Point(const _Point& pt) : x(pt.x), y(pt.y) { }
template<class _Tp> _Point<_Tp>& _Point<_Tp>::operator = (const _Point& pt) { x = pt.x; y = pt.y; }
template<class _Tp> _Tp _Point<_Tp>::dot(const _Point& pt) const { return (_Tp)x*pt.x + (_Tp)y*pt.y; }
template<class _Tp> inline double _Point<_Tp>::cross(const _Point<_Tp>& pt) const
{
	return ((double)x*pt.y - (double)y*pt.x);
}
typedef _Point<int>                 Point2i;
typedef _Point<double>              Point2f;
typedef _Point<float>               Point2d;
typedef _Point<int>                 Point;

/////////////////////////////////////////_Point3////////////////////////////////////////////
template<class _Tp> class _Point3
{
public:
	_Point3();
	_Point3(_Tp _x, _Tp _y, _Tp _z);
	_Point3(const _Point3& pt);

	_Point3& operator = (const _Point3& pt);

    _Tp dot(const _Point3& pt) const;
	_Point3 cross(const _Point3& pt) const;

    _Tp x, y, z;
};
template<class _Tp> inline _Point3<_Tp>::_Point3() : x(0), y(0), z(0) { }
template<class _Tp> inline _Point3<_Tp>::_Point3(_Tp _x, _Tp _y, _Tp _z) : x(_x), y(_y), z(_z) { }
template<class _Tp> inline _Point3<_Tp>::_Point3(const _Point3& pt) : x(pt.x), y(pt.y), z(pt.z) { }
template<class _Tp> _Point3<_Tp>& _Point3<_Tp>::operator = (const _Point3& pt) { x = pt.x; y = pt.y; z = pt.z; }
template<class _Tp> _Tp _Point3<_Tp>::dot(const _Point3& pt) const { return x*pt.x + y*pt.y + z*pt.z; }
template<typename _Tp> inline _Point3<_Tp> _Point3<_Tp>::cross(const _Point3<_Tp>& pt) const
{
	return Point3_<_Tp>(y*pt.z - z*pt.y, z*pt.x - x*pt.z, x*pt.y - y*pt.x);
}


typedef _Point3<int>                 Point3i;
typedef _Point3<float>               Point3f;
typedef _Point3<double>              Point3d;

///////////////////////////////////////_Rect//////////////////////////////////////////////
template<class _Tp> class _Rect
{
public:
	_Rect();
	_Rect(_Tp _x, _Tp _y, _Tp _width, _Tp _height);
	_Rect(const _Rect& r);
	_Rect(const _Point<_Tp>& org, const _Size<_Tp>& sz);
	_Rect(const _Point<_Tp>& pt1, const _Point<_Tp>& pt2);

	_Rect& operator = (const _Rect& r);
	_Point<_Tp> getTL() const { return _Point<_tp>(x, y); }                       // 上左
	_Point<_Tp> getBR() const { return _Point<_Tp>(x + width, y + height); }      // 下右

	_Size<_Tp> size() const { return _Size<_Tp>(width, heigt); }                  // 矩形的大小

	// 检查点是否在范围内
	bool contains(const _Point<_Tp>& pt) const { return (x < pt.x && (x + width) > pt.x && y < pt.y && (y + height) > pt.y); }

	_Tp x, y, width, height;
};
template<class _Tp> inline _Rect<_Tp>::_Rect() : x(0), y(0), width(0), height(0) { }
template<class _Tp> inline _Rect<_Tp>::_Rect(const _Rect& r) : x(r.x), y(r.y), width(r.width), height(r.height) { }
template<class _Tp> inline _Rect<_Tp>::_Rect(_Tp _x, _Tp _y, _Tp _width, _Tp _height) : x(_x), y(_y), width(_width), height(_height) { }
template<class _Tp> inline _Rect<_Tp>::_Rect(const _Point<_Tp>& org, const _Size<_Tp>& sz) :
	x(org.x), y(org.y),width(sz.width), height(_height){}
template<class _Tp> inline _Rect<_Tp>::_Rect(const _Point<_Tp>& pt1, const _Point<_Tp>& pt2)
{
	if (pt1.x < pt2.x) {
		x = pt1.x;
		width = pt2.x - pt1.x;
	}
	else {
		x = pt2.x;
		width = pt1.x - pt2.x;
	}

	if (pt1.y < pt2.y) {
		y = pt1.y;
		height = pt2.y - pt1.y;
	}
	else {
		y = pt2.y;
		height = pt1.y - pt2.y;
	}
}

template<class _Tp> inline _Rect<_Tp>& _Rect<_Tp>:: operator = (const _Rect& r)
{
	x = r.x;
	y = r.y;
	width = r.width;
	height = r.height;
}
typedef _Rect<int>                   Rect32s;
typedef _Rect<int>                   Rect;
typedef _Rect<float>                 Rect32f;
typedef _Rect<double>                Rect64f;


///////////////////////////////////////_Scalar//////////////////////////////////////////////
template<class _Tp> class _Scalar
{
public:
	_Scalar();
	_Scalar(_Tp _v0, _Tp _v1, _Tp _v2 = 0, _Tp _v3 = 0);
	_Scalar(_Tp _v0);
	_Scalar(const _Scalar& sr);

	_Scalar<_Tp> init(_Tp _v0);             // 全部初始化为v0
	_Scalar<_Tp> conj() const;              // 共轭
	bool isReal() const;                    // 是否为实数

private:
	_Tp v[4];
};
template<class _Tp> inline _Scalar<_Tp>::_Scalar() :v[0](0), v[1](0), v[2](0), v[3](0) { }
template<class _Tp> inline _Scalar<_Tp>::_Scalar(_Tp _v0) : v[0](_v0), v[1](0), v[2](0), v[3](0) { }
template<class _Tp> inline _Scalar<_Tp>::_Scalar(_Tp _v0, _Tp _v1, _Tp _v2 = 0, _Tp _v3 = 0):
	v[0](_v0), v[1](_v1), v[2](_v2), v[3](_v3){}
template<class _Tp> inline _Scalar<_Tp>::_Scalar(const _Scalar& sr) : v[0](sr.v[0]), v[1](sr.v[1]), v[2](sr.v[2]), v[3](sr.v[3]) { }
template<class _Tp> inline _Scalar<_Tp> _Scalar<_Tp>::init(_Tp _v0) { v[0] = v[1] = v[2] = v[3] = _v0; }
template<class _Tp> _Scalar<_Tp> _Scalar<_Tp>::conj() const { return _Scalar<_Tp>(v[0], -v[1], -v[2], -v[3]); }
template<class _Tp> bool _Scalar<_Tp>::isReal() const { return (v[1] == 0 && v[2] == 0 && v[3] == 0); }


typedef _Scalar<unsigned char>     Scalar;
typedef _Scalar<unsigned char>     Scalar8u;
typedef _Scalar<signed char>       Scalar8s;
typedef _Scalar<unsigned int>      Scalar32u;
typedef _Scalar<signed int>        Scalar32s;
typedef _Scalar<float>             Scalar32f;
typedef _Scalar<double>            Scalar64f;

/////////////////////////////////////////////_Size/////////////////////////////////////////////
template<class _Tp> class _Size
{
public:
	typedef _Tp value_type;

	//! various constructors
	_Size();
	_Size(_Tp _width, _Tp _height);
	_Size(const _Size& sz);

	_Size& operator = (const _Size& sz);
	_Tp area() const;

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


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Matrix8u Mat2Matrix8u(cv::Mat & mat);
template <class _type> void cvtColor(_Matrix<_type>&src, _Matrix<_type>&dst, int code);

// 多通道分离和混合
template <class _type> void spilt(_Matrix<_type> & src, std::vector<_Matrix<_type>> & mv);
template <class _type> void merge(_Matrix<_type> & src1, _Matrix<_type> & src2, _Matrix<_type> & dst);
template <class _type> void merge(std::vector<_Matrix<_type>> & src, _Matrix<_type> & dst);

// 离散傅里叶
void _dft(Matrix & src, Matrix & dst);
void dft(Matrix8u & src, Matrix & dst);

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