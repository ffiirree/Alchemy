#ifndef _ZCORE_TYPES_HPP
#define _ZCORE_TYPES_HPP


#ifndef __cplusplus
#  error types.hpp header must be compiled as C++
#endif

#include <assert.h>
#include "saturate.hpp"

namespace z
{

//////////////////////////////////////////////////////////////////////////
template<typename _Tp, int n> _Vec<_Tp, n>::_Vec()
{
    for (int i = 0; i < n; ++i)
        data_[i] = _Tp(0);
}
template<typename _Tp, int n> _Vec<_Tp, n>::_Vec(_Tp v0)
{
    assert(n >= 1);
    data_[0] = v0;
    for (int i = 1; i < n; ++i) data_[i] = _Tp(0);
}
template<typename _Tp, int n> _Vec<_Tp, n>::_Vec(_Tp v0, _Tp v1)
{
    assert(n >= 2);
    data_[0] = v0, data_[1] = v1;
    for (int i = 2; i < n; ++i) data_[i] = _Tp(0);
}
template<typename _Tp, int n> _Vec<_Tp, n>::_Vec(_Tp v0, _Tp v1, _Tp v2)
{
    assert(n >= 3);
    data_[0] = v0, data_[1] = v1, data_[2] = v2;
    for (int i = 3; i < n; ++i) data_[i] = _Tp(0);
}
template<typename _Tp, int n> _Vec<_Tp, n>::_Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3)
{
    assert(n >= 4);
    data_[0] = v0, data_[1] = v1, data_[2] = v2, data_[3] = v3;
    for (int i = 4; i < n; ++i) data_[i] = _Tp(0);
}

template<typename _Tp, int n> _Vec<_Tp, n>::_Vec(const _Tp* vals)
{
    for (int i = 0; i < n; ++i)
        data_[i] = vals[i];
}

template<typename _Tp, int n> _Vec<_Tp, n>::_Vec(const _Vec<_Tp, n>&v) :_Vec<_Tp, n>(v.data_) {  }

template<typename _Tp, int n> _Vec<_Tp, n>& _Vec<_Tp, n >::operator = (std::initializer_list<_Tp> list)
{
    assert(list.size() >= n);

    int idx = 0;
    for (const auto& i : list)
        data_[idx++] = i;

    for (; idx < n; ++idx)
        data_[idx] = _Tp(0);

    return *this;
}

template<typename _Tp, int n>
const _Tp& _Vec<_Tp, n >::operator[](int i) const
{
    assert((static_cast<unsigned>(i) < static_cast<unsigned>(n)));
    return data_[i];
}

template<typename _Tp, int n>
_Tp& _Vec<_Tp, n >::operator[](int i)
{
    assert((static_cast<unsigned>(i) < static_cast<unsigned>(n)));
    return data_[i];
}

template<typename _Tp, int n>
const _Tp& _Vec<_Tp, n >::operator()(int i) const
{
    assert((static_cast<unsigned>(i) < static_cast<unsigned>(n)));
    return data_[i];
}

template<typename _Tp, int n>
_Tp& _Vec<_Tp, n >::operator()(int i)
{
    assert(static_cast<unsigned>(i) < static_cast<unsigned>(n));
    return data_[i];
}

template<typename _Tp, int n>
_Vec<_Tp, n> _Vec<_Tp, n>::all(_Tp val)
{
    _Vec<_Tp, n> v;
    for (int i = 0; i < n; ++i)
        v.data_[i] = val;
    return v;
}

template<typename _Tp, int n>
_Vec<_Tp, n> _Vec<_Tp, n >::zeros()
{
    return all(0);
}
template<typename _Tp, int n>
_Vec<_Tp, n> _Vec<_Tp, n >::ones()
{
    return all(1);
}

/////////////////////////////////////////_Scalar////////////////////////////////////////////
template<class _Tp> _Scalar<_Tp>::_Scalar() { v[0] = v[1] = v[2] = v[3] = 0; }
template<class _Tp> _Scalar<_Tp>::_Scalar(_Tp _v0) { v[0] = _v0; v[1] = v[2] = v[3] = 0; }
template<class _Tp> _Scalar<_Tp>::_Scalar(_Tp _v0, _Tp _v1, _Tp _v2, _Tp _v3)
{
    v[0] = _v0;
    v[1] = _v1;
    v[2] = _v2;
    v[3] = _v3;
}

template<class _Tp> void _Scalar<_Tp>::init(_Tp _v0) { v[0] = v[1] = v[2] = v[3] = _v0; }
template<class _Tp> _Scalar<_Tp> _Scalar<_Tp>::conj() const { return _Scalar<_Tp>(v[0], -v[1], -v[2], -v[3]); }
template<class _Tp> bool _Scalar<_Tp>::isReal() const { return (v[1] == 0 && v[2] == 0 && v[3] == 0); }


template <typename _Tp>
_Scalar<_Tp> z::operator==(const _Scalar<_Tp>& a, const _Scalar<_Tp>& b)
{
    return a.v[0] == b.v[0] && a.v[1] == b.v[1] && a.v[2] == b.v[2] && a.v[3] == b.v[3];
}

template <typename _Tp>
_Scalar<_Tp> z::operator!=(const _Scalar<_Tp>& a, const _Scalar<_Tp>& b)
{
    return !(a == b);
}

template<typename _Tp>
_Scalar<_Tp> z::operator += (const _Scalar<_Tp>& a, const _Scalar<_Tp>& b)
{
    a.v[0] += b.v[0];
    a.v[1] += b.v[1];
    a.v[2] += b.v[2];
    a.v[3] += b.v[3];
    return a;
}


template <typename _Tp>
_Scalar<_Tp> z::operator -= (const _Scalar<_Tp>& a, const _Scalar<_Tp>& b)
{
    a.v[0] -= b.v[0];
    a.v[1] -= b.v[1];
    a.v[2] -= b.v[2];
    a.v[3] -= b.v[3];
    return a;
}

template<typename _Tp>
_Scalar<_Tp> operator *= (const _Scalar<_Tp>& a, _Tp v)
{
    a.v[0] *= v;
    a.v[1] *= v;
    a.v[2] *= v;
    a.v[3] *= v;
    return a;
}

template<typename _Tp>
_Scalar<_Tp> operator + (const _Scalar<_Tp>& a, const _Scalar<_Tp>& b)
{
    // Auto cast
    return z::_Scalar<_Tp>(
        a.v[0] + b.v[0],
        a.v[1] + b.v[1],
        a.v[2] + b.v[2],
        a.v[3] + b.v[3]);
}


template <typename _Tp>
_Scalar<_Tp> operator - (const _Scalar<_Tp>& a, const _Scalar<_Tp>& b)
{
    return _Scalar<_Tp>(
        saturate_cast<_Tp>(a.v[0] - b.v[0]),
        saturate_cast<_Tp>(a.v[1] - b.v[1]),
        saturate_cast<_Tp>(a.v[2] - b.v[2]),
        saturate_cast<_Tp>(a.v[3] - b.v[3]));
}

template<typename _Tp>
_Scalar<_Tp> operator * (const _Scalar<_Tp>& a, _Tp v)
{
    return _Scalar<_Tp>(a.v[0] * v,
        a.v[1] * v,
        a.v[2] * v,
        a.v[3] * v);
}


template<typename _Tp>
_Scalar<_Tp> operator * (_Tp v, const _Scalar<_Tp>& a)
{
    return a * v;
}

template<typename _Tp>
_Scalar<_Tp> operator - (const _Scalar<_Tp>& s)
{
    return _Scalar<_Tp>(
        saturate_cast<_Tp>(-s.v[0]),
        saturate_cast<_Tp>(-s.v[1]),
        saturate_cast<_Tp>(-s.v[2]),
        saturate_cast<_Tp>(-s.v[3]));
}

};

#endif // !_ZCORE_TYPES_HPP