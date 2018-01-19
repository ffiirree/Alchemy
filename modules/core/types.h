#ifndef ALCHEMY_CORE_TYPES_H
#define ALCHEMY_CORE_TYPES_H

#include <initializer_list>
#include <complex>
#include "traits.h"

namespace alchemy
{
/////////////////////////////////////////_Vec////////////////////////////////////////////
template <class _Tp, int n>
class _Vec {
public:
    typedef _Tp value_type;

    _Vec();
    _Vec(_Tp v0);
    _Vec(_Tp v0, _Tp v1);
    _Vec(_Tp v0, _Tp v1, _Tp v2);
    _Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3);
    _Vec(const _Tp* vals);

    _Vec(const _Vec<_Tp, n>&v);
    _Vec<_Tp, n>& operator = (std::initializer_list<_Tp> list);

    static _Vec all(_Tp val);
    static _Vec zeros();
    static _Vec ones();

    // element access
    const _Tp& operator[](int i) const;
    _Tp &operator[](int i);
    const _Tp& operator()(int i) const;
    _Tp& operator()(int i);

    _Tp data_[n];
};

template <class _Tp, int n> bool operator ==(const _Vec<_Tp, n> & v1, const _Vec<_Tp, n> &v2);
template <class _Tp, int n> bool operator !=(const _Vec<_Tp, n> & v1, const _Vec<_Tp, n> &v2);
template <class _Tp, int n> std::ostream & operator <<(std::ostream & os, const _Vec<_Tp, n> & c);

typedef _Vec<uint8_t, 2> Vec2u8;
typedef _Vec<uint8_t, 3> Vec3u8;
typedef _Vec<uint8_t, 4> Vec4u8;

typedef _Vec<int8_t, 2> Vec2s8;
typedef _Vec<int8_t, 3> Vec3s8;
typedef _Vec<int8_t, 4> Vec4s8;

typedef _Vec<uint16_t, 2> Vec2u16;
typedef _Vec<uint16_t, 3> Vec3u16;
typedef _Vec<uint16_t, 4> Vec4u16;

typedef _Vec<int16_t, 2> Vec2s16;
typedef _Vec<int16_t, 3> Vec3s16;
typedef _Vec<int16_t, 4> Vec4s16;

typedef _Vec<float, 2> Vec2f32;
typedef _Vec<float, 3> Vec3f32;
typedef _Vec<float, 4> Vec4f32;

typedef _Vec<double, 2> Vec2f64;
typedef _Vec<double, 3> Vec3f64;
typedef _Vec<double, 4> Vec4f64;


template <typename _Tp, int n>
struct DataType<_Vec<_Tp, n>> {
    enum {
        depth = DataType<_Tp>::depth,
        channels = n,
        value = depth << 8 | n
    };
};

/////////////////////////////////////////////_Size/////////////////////////////////////////////
template<class _Tp>
class _Size
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

typedef _Size<int>      Size2i;
typedef _Size<double>   Size2d;
typedef _Size<float>    Sizef;
typedef _Size<int>      Size;

template <typename _Tp>
struct DataType<_Size<_Tp>> {
    enum {
        depth = DataType<_Tp>::depth,
        channels = 2,
        value = depth << 8 | channels
    };
};

template <typename _T> bool operator==(const _Size<_T> &p1, const _Size<_T> &p2) { return p1.width == p2.width && p1.height == p2.height; }
template <typename _T> bool operator!=(const _Size<_T> &p1, const _Size<_T> &p2) { return !(p1 == p2); }

/////////////////////////////////////////_Complex2////////////////////////////////////////////
// '2' make gcc happy.(C99 & C++11: _Complex)
template <typename _Tp>
using _Complex2 = std::complex<_Tp>;

using Complex8s = _Complex2<int8_t>;
using Complex8u = _Complex2<uint8_t>;
using Complex16s = _Complex2<int16_t>;
using Complex16u = _Complex2<uint16_t>;
using Complex32s = _Complex2<int32_t>;
using Complex32u = _Complex2<uint32_t>;
using Complex32f = _Complex2<float>;
using Complex64f = _Complex2<double>;
using Complex = _Complex2<double>;

template <typename _Tp>
struct DataType<_Complex2<_Tp>> {
    enum {
        depth = DataType<_Tp>::depth,
        channels = 2,
        value = depth << 8 | channels
    };
};

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

typedef _Point<int>                 Point2i;
typedef _Point<double>              Point2f;
typedef _Point<float>               Point2d;
typedef _Point<int>                 Point;

template <typename _Tp>
struct DataType<_Point<_Tp>> {
    enum {
        depth = DataType<_Tp>::depth,
        channels = 2,
        value = depth << 8 | channels
    };
};

// ！没有做溢出处理，之后加上，类似opencv的saturate_cast函数
template <typename _T> _Point<_T> operator+(const _Point<_T> &p1, const _Point<_T> &p2) { return{ p1.x + p2.x, p1.y + p2.y }; }
template <typename _T> bool operator==(const _Point<_T> &p1, const _Point<_T> &p2) { return p1.x == p2.x && p1.y == p2.y; }
template <typename _T> bool operator!=(const _Point<_T> &p1, const _Point<_T> &p2) { return !(p1 == p2); }
template <typename _T> bool operator>(const _Point<_T> &p1, const _Point<_T> &p2) { return p1.x > p2.x && p1.y > p2.y; }
template <typename _T> bool operator>=(const _Point<_T> &p1, const _Point<_T> &p2) { return p1.x >= p2.x && p1.y >= p2.y; }
template <typename _T> bool operator<(const _Point<_T> &p1, const  _Point<_T> &p2) { return p1.x < p2.x && p1.y < p2.y; }
template <typename _T> bool operator<=(const _Point<_T> &p1, const  _Point<_T> &p2) { return p1.x <= p2.x && p1.y <= p2.y; }


template<typename _T> std::ostream &operator<<(std::ostream &os, _Point<_T> &p);

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

typedef _Point3<int>                 Point3i;
typedef _Point3<float>               Point3f;
typedef _Point3<double>              Point3d;

template <typename _Tp>
struct DataType<_Point3<_Tp>> {
    enum {
        depth = DataType<_Tp>::depth,
        channels = 3,
        value = depth << 8 | channels
    };
};

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

typedef _Rect<int>                   Rect32s;
typedef _Rect<int>                   Rect;
typedef _Rect<float>                 Rect32f;
typedef _Rect<double>                Rect64f;

template <typename _Tp>
struct DataType<_Rect<_Tp>> {
    enum {
        depth = DataType<_Tp>::depth,
        channels = 4,
        value = depth << 8 | channels
    };
};

///////////////////////////////////////_Scalar//////////////////////////////////////////////
template<class _Tp> class _Scalar
{
public:
    _Scalar();
    _Scalar(_Tp _v0, _Tp _v1, _Tp _v2 = 0, _Tp _v3 = 0);
    _Scalar(_Tp _v0);

    void init(_Tp _v0);                     // 全部初始化为v0
    _Scalar<_Tp> conj() const;              // 共轭
    bool isReal() const;                    // 是否为实数

    _Tp& operator[](int i0) { return v[i0]; }
    const _Tp& operator[](int i0) const { return v[i0]; }
    _Tp v[4];
};

typedef _Scalar<unsigned char>     Scalar8u;
typedef _Scalar<signed char>       Scalar8s;
typedef _Scalar<unsigned int>      Scalar32u;
typedef _Scalar<signed int>        Scalar32s;
typedef _Scalar<float>             Scalar32f;
typedef _Scalar<double>            Scalar64f;
typedef Scalar64f                  Scalar;

template <typename _Tp>
struct DataType<_Scalar<_Tp>> {
    enum {
        depth = DataType<_Tp>::depth,
        channels = 4,
        value = depth << 8 | channels
    };
};

template<typename _Tp> bool operator == (const _Scalar<_Tp>& a, const _Scalar<_Tp>& b);
template<typename _Tp> bool operator != (const _Scalar<_Tp>& a, const _Scalar<_Tp>& b);

template<typename _Tp> _Scalar<_Tp> operator += (const _Scalar<_Tp>& a, const _Scalar<_Tp>& b);
template<typename _Tp> _Scalar<_Tp> operator -= (const _Scalar<_Tp>& a, const _Scalar<_Tp>& b);
template<typename _Tp> _Scalar<_Tp> operator *= (const _Scalar<_Tp>& a, _Tp v);

template<typename _Tp> _Scalar<_Tp> operator + (const _Scalar<_Tp>& a, const _Scalar<_Tp>& b);
template<typename _Tp> _Scalar<_Tp> operator - (const _Scalar<_Tp>& a, const _Scalar<_Tp>& b);
template<typename _Tp> _Scalar<_Tp> operator * (const _Scalar<_Tp>& a, _Tp v);
template<typename _Tp> _Scalar<_Tp> operator * (_Tp v, const _Scalar<_Tp>& a);

template<typename _Tp> _Scalar<_Tp> operator - (const _Scalar<_Tp>& s);

template<typename _Tp> std::ostream &operator<<(std::ostream & os, const _Scalar<_Tp> &item);

///////////////////////////////////////_Scalar//////////////////////////////////////////////
struct MatrixShape
{
    int rows = 0;
    int cols = 0;
    int chs = 0;
};
inline bool operator==(const MatrixShape& s1, const MatrixShape& s2) { return s1.rows == s2.rows && s1.cols == s2.cols && s1.chs == s2.chs; }
inline bool operator!=(const MatrixShape& s1, const MatrixShape& s2) { return !(s1 == s2); }
} // ! namespace alchemy

#include "core/types.hpp"
#endif //! ALCHEMY_CORE_TYPES_H