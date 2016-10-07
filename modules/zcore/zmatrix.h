/**
 ******************************************************************************
 * @file    zmatrix.h
 * @author  zlq
 * @version V1.0
 * @date    2016.9.7
 * @brief   模板类_Matrix的定义
 ******************************************************************************
 * @attention
 *
 *
 ******************************************************************************
 */
#ifndef _ZMATRIX_H
#define _ZMATRIX_H

#include <stdint.h>
#include <iostream>
#include "config.h"

#if defined(OPENCV)
#include <opencv2\core.hpp>
#endif

#ifdef __cplusplus

namespace z{

template <class _type> class _Matrix;
template<class _Tp> class _Size;
template<class _Tp> class _Rect;

typedef _Matrix<double>             Matrix;
typedef _Matrix<double>             Matrix64f;
typedef _Matrix<float>              Matrix32f;
typedef _Matrix<signed int>         Matrix32s;
typedef _Matrix<unsigned int>       Matrix32u;
typedef _Matrix<signed short>       Matrix16s;
typedef _Matrix<unsigned short>     Matrix16u;
typedef _Matrix<signed char>        Matrix8s;
typedef _Matrix<unsigned char>      Matrix8u;

template <class _type> class _Matrix {
public:
	_Matrix();
	_Matrix(int rows, int cols);
	_Matrix(int rows, int cols,  int channelsNum);
	_Matrix(_Size<int> size);
	_Matrix(_Size<int> size, int channelsNum);
	_Matrix(const _Matrix<_type>& m);
	~_Matrix();

	//! allocates new matrix data unless the matrix already has specified size and type.
	// previous data is unreferenced if needed.
	void create(int _rows, int _cols, int _chs);
	//! pointer to the reference counter;
	// when matrix points to user-allocated data, the pointer is NULL
	int* refcount;
	void release();
	int refAdd(int *addr, int delta);

	template<typename _Tp2> operator _Matrix<_Tp2>() const;

	_Matrix<_type>& operator = (const _Matrix<_type>& m);
	_Matrix<_type>& operator = (std::initializer_list<_type>);
	_Matrix<_type>& operator += (const _Matrix<_type>& m);
	_Matrix<_type>& operator -= (const _Matrix<_type>& m);

	// 检查这两个函数是否达到了想要的目的
	inline _type* operator[](size_t n) { return data + n * step; }
	inline const _type* operator[](size_t n) const { return data + n * step; }


	//! returns pointer to (i0,i1) submatrix along the dimensions #0 and #1
	_type* ptr(int i0);
	const _type* ptr(int i0) const;

	_type* ptr(int i0, int i1);
	const _type* ptr(int i0, int i1) const;

	_Matrix<_type>& operator()(_type * InputArray, size_t size);
	_Matrix<_type>& operator()(_type * InputArray, int rows, int cols);

#if defined(OPENCV)
	// 类型转换
	operator cv::Mat() const;
#endif

	//! returns deep copy of the matrix, i.e. the data is copied
	_Matrix<_type> clone() const;
	void copyTo(_Matrix<_type> & outputMatrix) const;

	//! Matlab-style matrix initialization
	void zeros();
	void ones();
	void eye();
	void zeros(int rows, int cols);
	void ones(int rows, int cols);
	void eye(int rows, int cols);
	void init(_type _v);

	//! returns true if matrix data is NULL
	inline bool empty() const { return data == nullptr; }
	inline size_t size() const { return _size; }
	inline bool equalSize(const _Matrix<_type> & m) const { return (rows == m.rows && cols == m.cols && chs == m.chs); }

	_Matrix<_type> inv();                            // 逆
	_Matrix<_type> t();                              // 转置

	_type rank();                                    // 求秩
	double tr();                                     // 迹
	
	_Matrix<_type> dot(_Matrix<_type> &m);           // 点乘
	_Matrix<_type> cross(_Matrix<_type> &m);         // 叉积
	void conv(Matrix &kernel, _Matrix<_type>&dst, bool norm = false);

	inline int channels() { return chs; }

	void swap(int32_t i0, int32_t j0, int32_t i1, int32_t j1);
	
	int rows, cols;
	_type *data;
	_type *datastart, *dataend;
	int step;
	int chs;

private:
	size_t _size;

	

	void initEmpty();
};

template <class _type> std::ostream &operator<<(std::ostream & os, const _Matrix<_type> &item);

template <class _type> bool operator==(const _Matrix<_type> &m1, const _Matrix<_type> &m2);
template <class _type> bool operator!=(const _Matrix<_type> &m1, const _Matrix<_type> &m2);

template <class _type> _Matrix<_type> operator*(_Matrix<_type> &m1, _Matrix<_type> &m2);
template <class _type> _Matrix<_type> operator*(_Matrix<_type> &m, _type delta);
template <class _type> _Matrix<_type> operator*(_type delta, _Matrix<_type> &m);

template <class _type> _Matrix<_type> operator+(_Matrix<_type> &m1, _Matrix<_type> &m2);
template <class _type> _Matrix<_type> operator+(_Matrix<_type> &m, _type delta);
template <class _type> _Matrix<_type> operator+(_type delta, _Matrix<_type> &m);

template <class _type> _Matrix<_type> operator-(_Matrix<_type> &m1, _Matrix<_type> &m2);
template <class _type> _Matrix<_type> operator-(_Matrix<_type> &m, _type delta);
template <class _type> _Matrix<_type> operator-(_type delta, _Matrix<_type> &m);

template <class _type> void conv(_Matrix<_type> &src, _Matrix<_type> &dst, Matrix &core);

/////////////////////////////////////////_Complex_////////////////////////////////////////////
template <class _Tp> class _Complex_
{
public:
	_Complex_();
	_Complex_(_Tp _re, _Tp _im);
	_Complex_(const _Complex_ & c);
	_Complex_ &operator=(const _Complex_ &c);

	_Complex_<_Tp>& operator+=(const _Complex_<_Tp> & c);
	_Complex_<_Tp>& operator-=(const _Complex_<_Tp> & c);

	_Tp re, im;
};
template <class _Tp> bool operator ==(const _Complex_<_Tp> & c1, const _Complex_<_Tp> &c2);
template <class _Tp> bool operator !=(const _Complex_<_Tp> & c1, const _Complex_<_Tp> &c2);
template <class _Tp> _Complex_<_Tp> operator * (const _Complex_<_Tp> & c1, const _Complex_<_Tp> &c2);
template <class _Tp> _Complex_<_Tp> operator + (const _Complex_<_Tp> & c1, const _Complex_<_Tp> &c2);
template <class _Tp> _Complex_<_Tp> operator - (const _Complex_<_Tp> & c1, const _Complex_<_Tp> &c2);

template <class _Tp> std::ostream & operator <<(std::ostream & os, const _Complex_<_Tp> & c);


typedef _Complex_<signed char> Complex8s;
typedef _Complex_<signed int> Complex32s;
typedef _Complex_<float> Complex32f;
typedef _Complex_<double> Complex64f;
typedef Complex64f Complex;


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
	return _Point3<_Tp>(y*pt.z - z*pt.y, z*pt.x - x*pt.z, x*pt.y - y*pt.x);
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
	_Point<_Tp> getTL() const { return _Point<_Tp>(x, y); }                       // 上左
	_Point<_Tp> getBR() const { return _Point<_Tp>(x + width, y + height); }      // 下右

	_Size<_Tp> size() const { return _Size<_Tp>(width, height); }                  // 矩形的大小

																				  // 检查点是否在范围内
	bool contains(const _Point<_Tp>& pt) const { return (x < pt.x && (x + width) > pt.x && y < pt.y && (y + height) > pt.y); }

	_Tp x, y, width, height;
};
template<class _Tp> inline _Rect<_Tp>::_Rect() : x(0), y(0), width(0), height(0) { }
template<class _Tp> inline _Rect<_Tp>::_Rect(const _Rect& r) : x(r.x), y(r.y), width(r.width), height(r.height) { }
template<class _Tp> inline _Rect<_Tp>::_Rect(_Tp _x, _Tp _y, _Tp _width, _Tp _height) : x(_x), y(_y), width(_width), height(_height) { }
template<class _Tp> inline _Rect<_Tp>::_Rect(const _Point<_Tp>& org, const _Size<_Tp>& sz) :
	x(org.x), y(org.y), width(sz.width), height(sz.height) {}
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
	_Tp v[4];
};
template<class _Tp> inline _Scalar<_Tp>::_Scalar() { v[0] = v[1] = v[2] = v[3] = 0; }
template<class _Tp> inline _Scalar<_Tp>::_Scalar(_Tp _v0) { v[0] = _v0; v[1] = v[2] = v[3] = 0; }
template<class _Tp> inline _Scalar<_Tp>::_Scalar(_Tp _v0, _Tp _v1, _Tp _v2 = 0, _Tp _v3 = 0)
{
	v[0] = _v0;
	v[1] = _v1;
	v[2] = _v2;
	v[3] = _v3;
}
template<class _Tp> inline _Scalar<_Tp>::_Scalar(const _Scalar& sr)
{
	v[0] = sr.v[0];
	v[1] = sr.v[1];
	v[2] = sr.v[2];
	v[3] = sr.v[3];
}
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

}
#endif // !__cplusplus

#include "operations.hpp"

#endif  // !_ZMATRIX_H