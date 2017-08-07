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
#ifndef _ZCORE_ZMATRIX_H
#define _ZCORE_ZMATRIX_H

#include <stdint.h>
#include "config.h"

#if defined(OPENCV)
#include "opencv2/opencv.hpp"
#endif

namespace z {

    //////////////////////////////////////////////////////////////////////////////////////////
    template <typename T> struct MatrixType { enum { type = 0 }; };

    // 特化
    template <> struct MatrixType<uint8_t> { enum { type = 0 }; };
    template <> struct MatrixType<int8_t> { enum { type = 1 }; };
    template <> struct MatrixType<uint16_t> { enum { type = 2 }; };
    template <> struct MatrixType<int16_t> { enum { type = 3 }; };
    template <> struct MatrixType<int32_t> { enum { type = 4 }; };
    template <> struct MatrixType<float> { enum { type = 5 }; };
    template <> struct MatrixType<double> { enum { type = 6 }; };


    /////////////////////////////////////////_Vec////////////////////////////////////////////
    template <class _Tp, int n> class Vec_ {
    public:
        typedef _Tp value_type;

        Vec_();
	    explicit Vec_(_Tp v0);
        Vec_(_Tp v0, _Tp v1);
        Vec_(_Tp v0, _Tp v1, _Tp v2);
        Vec_(_Tp v0, _Tp v1, _Tp v2, _Tp v3);
	    explicit Vec_(const _Tp* vals);

        Vec_(const Vec_<_Tp, n>&v);

        Vec_<_Tp, n>& operator = (std::initializer_list<_Tp> list);

        static Vec_ all(_Tp val);
        static Vec_ zeros();
        static Vec_ ones();

        // element access
        const _Tp& operator[](int i) const;
        _Tp &operator[](int i);
        const _Tp& operator()(int i) const;
        _Tp& operator()(int i);

//        template<typename _T2> operator Vec_<_T2, n>() const;

        //int size_ = n;

        _Tp data_[n];
    };

    typedef Vec_<uint8_t, 2> Vec2u8;
    typedef Vec_<uint8_t, 3> Vec3u8;
    typedef Vec_<uint8_t, 4> Vec4u8;

    typedef Vec_<int8_t, 2> Vec2s8;
    typedef Vec_<int8_t, 3> Vec3s8;
    typedef Vec_<int8_t, 4> Vec4s8;

    typedef Vec_<uint16_t, 2> Vec2u16;
    typedef Vec_<uint16_t, 3> Vec3u16;
    typedef Vec_<uint16_t, 4> Vec4u16;

    typedef Vec_<int16_t, 2> Vec2s16;
    typedef Vec_<int16_t, 3> Vec3s16;
    typedef Vec_<int16_t, 4> Vec4s16;

    typedef Vec_<float, 2> Vec2f32;
    typedef Vec_<float, 3> Vec3f32;
    typedef Vec_<float, 4> Vec4f32;

    typedef Vec_<double, 2> Vec2f64;
    typedef Vec_<double, 3> Vec3f64;
    typedef Vec_<double, 4> Vec4f64;

    /////////////////////////////////////////////_Size/////////////////////////////////////////////
    template<class _Tp> class _Size
    {
    public:
        typedef _Tp value_Tp;

        //! various constructors
        _Size();
        _Size(_Tp _width, _Tp _height);
        _Size(const _Size& sz);

        _Size& operator = (const _Size& sz);
        _Tp area() const;

        _Tp width, height; // the width and the height
    };
    template<class _Tp> _Size<_Tp>::_Size() :width(0), height(0) {}
    template<class _Tp> _Size<_Tp>::_Size(_Tp _width, _Tp _height) : width(_width), height(_height) {}
    template<class _Tp> _Size<_Tp>::_Size(const _Size& sz) : width(sz.width), height(sz.height) {}
    template<class _Tp> _Tp _Size<_Tp>::area() const { return width * height; }


    typedef _Size<int>      Size2i;
    typedef _Size<double>   Size2d;
    typedef _Size<float>    Sizef;
    typedef _Size<int>      Size;

    /////////////////////////////////////////_Matrix////////////////////////////////////////////
    template <class _Tp> class _Matrix;
    template<class _Tp> class _Rect;

    typedef _Matrix<double>             Matrix64f;
    typedef _Matrix<float>              Matrix32f;
    typedef _Matrix<signed int>         Matrix32s;
    typedef _Matrix<unsigned int>       Matrix32u;
    typedef _Matrix<signed short>       Matrix16s;
    typedef _Matrix<unsigned short>     Matrix16u;
    typedef _Matrix<signed char>        Matrix8s;
    typedef _Matrix<unsigned char>      Matrix8u;
    typedef _Matrix<unsigned char>      Matrix;

    template <class _Tp> class _Matrix {
    public:
        _Matrix() { }
        _Matrix(int rows, int cols, int _chs = 1);
	    explicit _Matrix(Size size, int _chs = 1);
        _Matrix(const _Matrix<_Tp>& m);
        _Matrix<_Tp>& operator = (const _Matrix<_Tp>& m);
        ~_Matrix();

        //! allocates new matrix data unless the matrix already has specified size and type.
        // previous data is unreferenced if needed.
        void create(int _rows, int _cols, int _chs = 1);
        void create(Size size, int _chs = 1);
        //! pointer to the reference counter;
        // when matrix points to user-allocated data, the pointer is NULL
        int* refcount = nullptr;
	    /**
         * \brief 
         */
        void release();
        int refAdd(int *addr, int delta);

        template<typename _Tp2> operator _Matrix<_Tp2>() const;

        _Matrix<_Tp>& operator = (std::initializer_list<_Tp>);
        _Matrix<_Tp>& operator += (const _Matrix<_Tp>& m);
        _Matrix<_Tp>& operator -= (const _Matrix<_Tp>& m);

        // 检查这两个函数是否达到了想要的目的
        _Tp* operator[](size_t n) { assert(!(static_cast<unsigned>(n) >= static_cast<unsigned>(rows)));  return data + n * step; }
        const _Tp* operator[](size_t n) const { assert(!(static_cast<unsigned>(n) >= static_cast<unsigned>(rows))); return data + n * step; }


        //! returns pointer to (i0,i1) submatrix along the dimensions #0 and #1
        _Tp* ptr(int i0);
        const _Tp* ptr(int i0) const;

        _Tp* ptr(int i0, int i1);
        const _Tp* ptr(int i0, int i1) const;

        _Tp* ptr(int i0, int i1, int i2);
        const _Tp* ptr(int i0, int i1, int i2) const;

        template<typename _T2> _T2* ptr(int i0);
        template<typename _T2> const _T2* ptr(int i0) const; 

        template<typename _T2> _T2* ptr(int i0, int i1);
        template<typename _T2> const _T2* ptr(int i0, int i1) const;

        template<typename _T2> _T2* ptr(int i0, int i1, int i3);
        template<typename _T2> const _T2* ptr(int i0, int i1, int i3) const;


        _Matrix<_Tp>& operator()(_Tp * InputArray, z::Size size);
        _Matrix<_Tp>& operator()(_Tp * InputArray, int rows, int cols);

#if defined(OPENCV)
        // 类型转换
	    explicit operator cv::Mat() const;
#endif

        //! returns deep copy of the matrix, i.e. the data is copied
        _Matrix<_Tp> clone() const;
        void copyTo(_Matrix<_Tp> & outputMatrix) const;

        //! Matlab-style matrix initialization
        void zeros();
        void ones();
        void eye();
        void zeros(int rows, int cols);
        void ones(int rows, int cols);
        void eye(int rows, int cols);
        void init(_Tp _v);

        //! returns true if matrix data is NULL
	    bool empty() const { return data == nullptr; }
	    Size size() const { return{ cols, rows }; }
	    bool equalSize(const _Matrix<_Tp> & m) const { return (rows == m.rows && cols == m.cols && chs == m.chs); }

        _Matrix<_Tp> inv();                             // 逆
        _Matrix<_Tp> t();                               // 转置

        _Tp rank();                                     // 求秩
        double tr();                                    // 迹

        _Matrix<_Tp> dot(_Matrix<_Tp> &m);              // 点乘
        _Matrix<_Tp> cross(_Matrix<_Tp> &m);            // 叉积
        template <typename _T2> _Matrix<_Tp> conv(const _Matrix<_T2> &kernel, bool norm = false) const;

	    int channels() const { return chs; }

        void swap(int32_t i0, int32_t j0, int32_t i1, int32_t j1);

        int flags = 0;

        // 每个通道元素的字节数
        int chssize = 0;

        int rows = 0;
        int cols = 0;
        _Tp *data = nullptr;
        _Tp *datastart = nullptr;
        _Tp *dataend = nullptr;
        int step = 0;
        int chs = 0;
        size_t size_ = 0;
    };

    template <class _Tp> std::ostream &operator<<(std::ostream & os, const _Matrix<_Tp> &item);

    template <class _Tp> bool operator==(const _Matrix<_Tp> &m1, const _Matrix<_Tp> &m2);
    template <class _Tp> bool operator!=(const _Matrix<_Tp> &m1, const _Matrix<_Tp> &m2);

    template <class _Tp> _Matrix<_Tp> operator*(_Matrix<_Tp> &m1, _Matrix<_Tp> &m2);
    template <class _Tp> _Matrix<_Tp> operator*(_Matrix<_Tp> &m, _Tp delta);
    template <class _Tp> _Matrix<_Tp> operator*(_Tp delta, _Matrix<_Tp> &m);

    template <class _Tp> _Matrix<_Tp> operator+(_Matrix<_Tp> &m1, _Matrix<_Tp> &m2);
    template <class _Tp> _Matrix<_Tp> operator+(_Matrix<_Tp> &m, _Tp delta);
    template <class _Tp> _Matrix<_Tp> operator+(_Tp delta, _Matrix<_Tp> &m);

    template <class _Tp> _Matrix<_Tp> operator-(_Matrix<_Tp> &m1, _Matrix<_Tp> &m2);
    template <class _Tp> _Matrix<_Tp> operator-(_Matrix<_Tp> &m, _Tp delta);
    template <class _Tp> _Matrix<_Tp> operator-(_Tp delta, _Matrix<_Tp> &m);

    template <class _T1, class _T2> _Matrix<_T1> operator>(const _Matrix<_T1> &m, _T2 threshold);
    template <class _T1, class _T2> _Matrix<_T1> operator<(const _Matrix<_T1> &m, _T2 threshold);
    template <class _T1, class _T2> _Matrix<_T1> operator>=(const _Matrix<_T1> &m, _T2 threshold);
    template <class _T1, class _T2> _Matrix<_T1> operator<=(const _Matrix<_T1> &m, _T2 threshold);

    template <class _Tp> void conv(_Matrix<_Tp> &src, _Matrix<_Tp> &dst, Matrix64f &core);

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

        _Point<_Tp>& operator = (const _Point<_Tp>& pt);

        _Tp dot(const _Point& pt) const;                    // 点乘
        double cross(const _Point& pt) const;               // 叉积
//        bool inside(const _Rect<_Tp>& r) const;             // 检查点是否在区域内

        _Tp x, y;
    };
    template<class _Tp> _Point<_Tp>::_Point() : x(0), y(0) { }
    template<class _Tp> _Point<_Tp>::_Point(_Tp _x, _Tp _y) : x(_x), y(_y) { }
    template<class _Tp> _Point<_Tp>::_Point(const _Point& pt) : x(pt.x), y(pt.y) { }
    template<class _Tp> _Point<_Tp>& _Point<_Tp>::operator = (const _Point& pt) { x = pt.x; y = pt.y; return *this; }
    template<class _Tp> _Tp _Point<_Tp>::dot(const _Point& pt) const { return static_cast<_Tp>(x)*pt.x + static_cast<_Tp>(y)*pt.y; }
    template<class _Tp> double _Point<_Tp>::cross(const _Point<_Tp>& pt) const
    {
        return (static_cast<double>(x)*pt.y - static_cast<double>(y)*pt.x);
    }
    typedef _Point<int>                 Point2i;
    typedef _Point<double>              Point2f;
    typedef _Point<float>               Point2d;
    typedef _Point<int>                 Point;

    // ！没有做溢出处理，之后加上，类似opencv的saturate_cast函数
    template <typename _T> _Point<_T> operator+(_Point<_T> &p1, _Point<_T> &p2) { return{ p1.x + p2.x, p1.y + p2.y }; }
    template <typename _T> bool operator==(_Point<_T> &p1, _Point<_T> &p2) { return p1.x == p2.x && p1.y == p2.y; }
    template <typename _T> bool operator!=(_Point<_T> &p1, _Point<_T> &p2) { return !(p1 == p2); }
    template <typename _T> bool operator>(_Point<_T> &p1, _Point<_T> &p2) { return p1.x > p2.x && p1.y > p2.y; }
    template <typename _T> bool operator>=(_Point<_T> &p1, _Point<_T> &p2) { return p1.x >= p2.x && p1.y >= p2.y; }
    template <typename _T> bool operator<(_Point<_T> &p1, _Point<_T> &p2) { return p1.x < p2.x && p1.y < p2.y; }
    template <typename _T> bool operator<=(_Point<_T> &p1, _Point<_T> &p2) { return p1.x <= p2.x && p1.y <= p2.y; }


    template<typename _T> std::ostream &operator<<(std::ostream &os, _Point<_T> &p)
    {
        if (sizeof(_T) == 1)
            os << "[" << static_cast<int>(p.x) << ", " << static_cast<int>(p.y) << "]";
        else
            os << "[" << p.x << ", " << p.y << "]";

        return os;
    }

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
    template<class _Tp> _Point3<_Tp>::_Point3() : x(0), y(0), z(0) { }
    template<class _Tp> _Point3<_Tp>::_Point3(_Tp _x, _Tp _y, _Tp _z) : x(_x), y(_y), z(_z) { }
    template<class _Tp> _Point3<_Tp>::_Point3(const _Point3& pt) : x(pt.x), y(pt.y), z(pt.z) { }
	template<class _Tp> _Point3<_Tp>& _Point3<_Tp>::operator = (const _Point3& pt) { x = pt.x; y = pt.y; z = pt.z; return *this; }
    template<class _Tp> _Tp _Point3<_Tp>::dot(const _Point3& pt) const { return x*pt.x + y*pt.y + z*pt.z; }
    template<typename _Tp> _Point3<_Tp> _Point3<_Tp>::cross(const _Point3<_Tp>& pt) const
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
    template<class _Tp> _Rect<_Tp>::_Rect() : x(0), y(0), width(0), height(0) { }
    template<class _Tp> _Rect<_Tp>::_Rect(const _Rect& r) : x(r.x), y(r.y), width(r.width), height(r.height) { }
    template<class _Tp> _Rect<_Tp>::_Rect(_Tp _x, _Tp _y, _Tp _width, _Tp _height) : x(_x), y(_y), width(_width), height(_height) { }
    template<class _Tp> _Rect<_Tp>::_Rect(const _Point<_Tp>& org, const _Size<_Tp>& sz) :
        x(org.x), y(org.y), width(sz.width), height(sz.height) {}
    template<class _Tp> _Rect<_Tp>::_Rect(const _Point<_Tp>& pt1, const _Point<_Tp>& pt2)
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

    template<class _Tp> _Rect<_Tp>& _Rect<_Tp>:: operator = (const _Rect& r)
    {
        x = r.x;
        y = r.y;
        width = r.width;
        height = r.height;
	    return *this;
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

        void init(_Tp _v0);             // 全部初始化为v0
        _Scalar<_Tp> conj() const;              // 共轭
        bool isReal() const;                    // 是否为实数
        _Tp v[4];
    };
    template<class _Tp> _Scalar<_Tp>::_Scalar() { v[0] = v[1] = v[2] = v[3] = 0; }
    template<class _Tp> _Scalar<_Tp>::_Scalar(_Tp _v0) { v[0] = _v0; v[1] = v[2] = v[3] = 0; }
    template<class _Tp> _Scalar<_Tp>::_Scalar(_Tp _v0, _Tp _v1, _Tp _v2, _Tp _v3)
    {
        v[0] = _v0;
        v[1] = _v1;
        v[2] = _v2;
        v[3] = _v3;
    }
    template<class _Tp> _Scalar<_Tp>::_Scalar(const _Scalar& sr)
    {
        v[0] = sr.v[0];
        v[1] = sr.v[1];
        v[2] = sr.v[2];
        v[3] = sr.v[3];
    }
    template<class _Tp> void _Scalar<_Tp>::init(_Tp _v0) { v[0] = v[1] = v[2] = v[3] = _v0; }
    template<class _Tp> _Scalar<_Tp> _Scalar<_Tp>::conj() const { return _Scalar<_Tp>(v[0], -v[1], -v[2], -v[3]); }
    template<class _Tp> bool _Scalar<_Tp>::isReal() const { return (v[1] == 0 && v[2] == 0 && v[3] == 0); }


    typedef _Scalar<unsigned char>     Scalar;
    typedef _Scalar<unsigned char>     Scalar8u;
    typedef _Scalar<signed char>       Scalar8s;
    typedef _Scalar<unsigned int>      Scalar32u;
    typedef _Scalar<signed int>        Scalar32s;
    typedef _Scalar<float>             Scalar32f;
    typedef _Scalar<double>            Scalar64f;

    

}

#include "operations.hpp"

#endif  // !_ZCORE_ZMATRIX_H