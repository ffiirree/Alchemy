/**
 ******************************************************************************
 * @file    operations.hpp
 * @author  zlq
 * @version V1.0
 * @date    2016.9.7
 * @brief   模板类_Matrix的实现
 ******************************************************************************
 * @attention
 *
 *
 ******************************************************************************
 */
#ifndef _OPERATIONS_HPP
#define _OPERATIONS_HPP

#include <algorithm>
#include <limits>
#include "saturate.hpp"
#include "debug.h"
#include "util.h"

// 不使用任何宏定义的max和min
#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

#define ZMATRIX_CONTINUOUS_MASK (1 << 16)
#define ZMATRIX_CH_MASK         (0x000000ff)
#define ZMATRIX_TYPE_MASK       (0x0000ffff)
#define ZMATRIX_DEPTH_MASK      (0x0000ff00)

namespace z {

/////////////////////////////////////////////////////////////////////////////////////////////
template <typename _Tp>
void _Matrix<_Tp>::create(int _rows, int _cols, int _chs)
{
    flags = MatrixType<_Tp>::depth << 8 | _chs;
    esize_ = sizeof(_Tp) * _chs;

    rows = _rows;
    cols = _cols;
    step = _cols * esize_;
    size_ = _rows * _cols;

    // Free the memary if the reference counter is 1.
    release();

    // Alloc memory.
    datastart = data = new uint8_t[size_ * esize_];
    dataend = data + size_ * esize_;

    // Reference counter.
    refcount = new int(1);
}

template <typename _Tp>
void _Matrix<_Tp> ::create(Size size, int _chs)
{
    create(size.height, size.width, _chs);
}

template <typename _Tp>
int _Matrix<_Tp>::channels() const
{
    return flags & ZMATRIX_CH_MASK;
}

template <typename _Tp>
_Matrix<_Tp> _Matrix<_Tp>::operator()(const Rect& roi) const
{
    return _Matrix<_Tp>(*this, roi);
}

template <typename _Tp>
bool _Matrix<_Tp>::isContinuous() const
{
    return !(flags & ZMATRIX_CONTINUOUS_MASK);
}

template <typename _Tp>
int _Matrix<_Tp>::type() const
{
    return flags & ZMATRIX_TYPE_MASK;
}

template <class _Tp>
int _Matrix<_Tp>::depth() const
{
    return (flags & ZMATRIX_DEPTH_MASK) >> 8;
}

template <typename _Tp>
_Matrix<_Tp>::_Matrix(int _rows, int _cols, int _chs)
{
    create(_rows, _cols, _chs);
}

template <typename _Tp>
_Matrix<_Tp>::_Matrix(Size size, int _chs)
{
    create(size.height, size.width, _chs);
}

template <typename _Tp>
_Matrix<_Tp>::_Matrix(int rows, int cols, int _chs, const Scalar&s)
{
    create(rows, cols, _chs);
    *this = s;
}

template <typename _Tp>
_Matrix<_Tp>::_Matrix(Size size, int _chs, const Scalar&s)
{
    create(size, _chs);
    *this = s;
}

template <typename _Tp>
_Matrix<_Tp>& _Matrix<_Tp>::operator=(const _Tp& val)
{
    _Matrix<_Tp>::operator=(Scalar(val));
    return *this;
}

template <typename _Tp>
_Matrix<_Tp> _Matrix<_Tp>::reshape(int cn) const
{
    if (cn == 0 || cn == channels())
        return *this;

    auto width = cols * channels();
    assert(width % cn == 0);

    _Matrix<_Tp> res = *this;

    res.cols = width / cn;
    res.flags = (flags & ~ZMATRIX_CH_MASK) | cn;
    res.size_ = res.rows * res.cols;
    res.esize_ = sizeof(_Tp) * cn;

    return res;
}

template <typename _Tp>
_Matrix<_Tp>& _Matrix<_Tp>::operator=(const Scalar& s)
{
    assert(channels() <= 4);

    if (s[0] == s[1] && s[1] == s[2] && s[2] == s[3] && s[0] == 0) {
        if (isContinuous()) {
            memset(data, 0, size_ * esize_);
            return *this;
        }

        for (auto i = 0; i < rows; ++i) {
            memset(ptr(i), 0, cols * esize_);
        }
        return *this;
    }


    for (auto i = 0; i < channels() && i < 4; ++i) {
        (reinterpret_cast<_Tp*>(data))[i] = saturate_cast<_Tp>(s[i]);
    }

    for (auto i = 0; i < rows; ++i) {
        for (auto j = 0; j < cols; ++j) {
            auto dst = ptr(i, j);
            memcpy(dst, data, esize());
        }
    }

    return *this;
}

template <typename _Tp>
_Matrix<_Tp>::_Matrix(const _Matrix& m)
{
    *this = m;
}

template <typename _Tp>
template <typename _T2>
_Matrix<_Tp>::_Matrix(const _Matrix<_T2>& m)
{
    flags = (flags & ~ZMATRIX_TYPE_MASK) | MatrixType<_Tp>::type;
    *this = m;
}

template <typename _Tp>
_Matrix<_Tp>::_Matrix(const _Matrix<_Tp>& m, const Rect& roi)
    : flags(m.flags), rows(roi.height), cols(roi.width),
    datastart(m.datastart), dataend(m.dataend), step(m.step),
    esize_(m.esize_), refcount(m.refcount)
{
    assert(roi.x >= 0 && roi.y >= 0
        && roi.width >= 0 && roi.height >= 0
        && roi.width + roi.x <= m.cols
        && roi.height + roi.y <= m.rows);

    data = m.data + roi.y * step + roi.x * esize_;
    if (roi.width < m.cols)
        flags |= ZMATRIX_CONTINUOUS_MASK;
    if (roi.height == 1)
        flags &= ~ZMATRIX_CONTINUOUS_MASK;

    size_ = rows * cols;

    refAdd(refcount, 1);
}

template <typename _Tp>
_Matrix<_Tp>::_Matrix(std::initializer_list<_Tp> list)
{
    if (list.size() == 0) return;

    create(list.size(), 1, 1);
    *this = 0;

    auto begin = reinterpret_cast<_Tp *>(data);
    for (const auto&i : list) {
        *begin++ = i;
    }
}

template <typename _Tp>
_Matrix<_Tp>& _Matrix<_Tp>::operator=(const _Matrix&m)
{
    if (this != &m) {
        if (m.refcount)
            refAdd(m.refcount, 1);

        // 释放掉左值的内容
        release();

        // 赋值
        flags = m.flags;
        esize_ = m.esize_;
        size_ = m.size_;
        data = m.data;
        refcount = m.refcount;
        rows = m.rows;
        cols = m.cols;
        step = m.step;
        datastart = m.datastart;
        dataend = m.dataend;
    }

    return *this;
}

template <typename _Tp>
template <typename _T2>
_Matrix<_Tp>& _Matrix<_Tp>::operator=(const _Matrix<_T2> &m)
{
    // : _Matrix<int8_t>(3, 3, 3) => _Matrix<Vec3u8>(3, 3, 3);
    if (MatrixType<_Tp>::type == m.type()) {
        auto mptr = reinterpret_cast<const _Matrix<_Tp> *>(&m);
        _Matrix::operator=(*mptr);
        return *this;
    }

    // : _Matrix<int8_t>(3, 3, 3) = > _Matrix<Vec_<int8_t, 1>>(3, 9, 1);
    if (MatrixType<_Tp>::depth == m.depth()) {
        *this = m.reshape(MatrixType<_Tp>::channels);
        return *this;
    }

    // : _Matrix<int8_t>(3, 3, 3) = > _Matrix<float>(3, 3, 3);
    if (size() != m.size() || channels() != m.channels())
        create(m.size(), m.channels());

    for (auto i = 0; i < m.rows; ++i) {
        for (auto j = 0; j < m.cols; ++j) {
            for (auto k = 0; k < m.channels(); ++k) {
                this->at(i, j, k) = saturate_cast<_Tp>(m.at(i, j, k));
            }
        }
    }
    return *this;
}

template <typename _Tp>
_Matrix<_Tp>& _Matrix<_Tp>::operator=(std::initializer_list<_Tp> list)
{
    assert(total() > 0);

    if (list.size() == 0) return *this;

    auto count = std::min(total(), list.size());
    auto data_ptr = ptr<_Tp>();

    if (isContinuous()) {
        for (const auto&e : list) {
            *data_ptr++ = e;
        }
        return *this;
    }

    auto i = 0;
    for (const auto&e : list) {
        data_ptr = ptr<_Tp>(i / (cols * channels()));
        data_ptr[(i++) % (cols * channels())] = e;
    }
    return *this;
}

template <typename _Tp>
int _Matrix<_Tp>::refAdd(int *addr, int delta)
{
    auto temp = *addr;
    *addr += delta;
    return temp;
}

template <typename _Tp>
void _Matrix<_Tp>::release()
{
    if (refcount && refAdd(refcount, -1) == 1) {
        delete[] data;
        data = datastart = dataend = nullptr;

        delete refcount;
        refcount = nullptr;
    }
}

template <typename _Tp>
_Matrix<_Tp>::~_Matrix()
{
    release();
}

template <typename _Tp>
_Matrix<_Tp> _Matrix<_Tp>::zeros(int _rows, int _cols, int _chs)
{
    return _Matrix<_Tp>(_rows, _cols, _chs, Scalar(0));
}

template <typename _Tp>
_Matrix<_Tp> _Matrix<_Tp>::zeros(Size size, int _chs)
{
    return  _Matrix<_Tp>::zeros(size.height, size.width, _chs);
}

template <typename _Tp>
_Matrix<_Tp> _Matrix<_Tp>::ones(int _rows, int _cols, int _chs)
{
    _Matrix<_Tp> temp(_rows, _cols, _chs);
    for (auto i = 0; i < temp.channels(); ++i) {
        reinterpret_cast<_Tp *>(temp.data)[i] = 1;
    }
    for (auto i = 0; i < temp.total(); ++i) {
        memcpy(temp.data + i * temp.esize(), temp.data, temp.esize());
    }

    return temp;
}

template <typename _Tp>
_Matrix<_Tp> _Matrix<_Tp>::ones(Size size, int _chs)
{
    return _Matrix<_Tp>::ones(size.height, size.width, _chs);
}


template <typename _Tp>
_Matrix<_Tp> _Matrix<_Tp>::eye(int _rows, int _cols, int _chs)
{
    assert(_chs == 1 && _rows == _cols);

    _Matrix<_Tp> temp = _Matrix<_Tp>::zeros(_rows, _cols, _chs);

    for (auto i = 0; i < _rows; ++i) {
        temp.data[i * _cols + i] = _Tp(1);
    }
    return temp;
}

template <typename _Tp>
_Matrix<_Tp> _Matrix<_Tp>::eye(Size size, int _chs)
{
    return _Matrix<_Tp>::eye(size.height, size.width, _chs);
}

template <typename _Tp>
void _Matrix<_Tp>::copyTo(_Matrix<_Tp> & outputMatrix) const
{
    outputMatrix.create(rows, cols, channels());
    if (isContinuous() && outputMatrix.isContinuous()) {
        memcpy(outputMatrix.data, data, size_ * channels() * sizeof(_Tp));
        return;
    }

    for (auto i = 0; i < rows; ++i) {
        auto src = ptr(i);
        auto dst = outputMatrix.ptr(i);
        memcpy(dst, src, cols * esize_);
    }
}

template <typename _Tp>
_Matrix<_Tp> _Matrix<_Tp>::clone() const
{
    _Matrix<_Tp> m;
    copyTo(m);
    return m;
}

template <typename _Tp>
_Matrix<_Tp>& _Matrix<_Tp>::operator()(_Tp * InputArray, Size size)
{
    create(size.height, size.width, 1);
    for (size_t i = 0; i < size_; ++i)
        data[i] = InputArray[i];
    return *this;
}

template <typename _Tp>
_Matrix<_Tp>& _Matrix<_Tp>::operator()(_Tp * InputArray, int _rows, int _cols)
{
    return _Matrix<_Tp>::operator()(InputArray, Size(_cols, _rows));
}

#if defined(OPENCV)
/**
* @brief 向openCV中的Mat类转换
*/
template <typename _Tp>
_Matrix<_Tp>::operator cv::Mat() const
{
    cv::Mat temp(rows, cols, CV_MAKETYPE(flags, channels()));

    memcpy(temp.data, data, size_ * channels() * sizeof(_Tp));

    return temp;
}
#endif

template <typename _Tp>
_Tp* _Matrix<_Tp>::ptr(int row)
{
    assert(static_cast<unsigned>(row) < static_cast<unsigned>(rows));

    return reinterpret_cast<_Tp *>(data + row * step);
}


template <typename _Tp>
const _Tp* _Matrix<_Tp>::ptr(int row) const
{
    assert(static_cast<unsigned>(row) < static_cast<unsigned>(rows));

    return reinterpret_cast<_Tp *>(data + row * step);
}

template <typename _Tp>
_Tp* _Matrix<_Tp>::ptr(int row, int col)
{
    assert(!(static_cast<unsigned>(row) >= static_cast<unsigned>(rows)
        || static_cast<unsigned>(col) >= static_cast<unsigned>(cols)));

    return reinterpret_cast<_Tp *>(data + row * step + col * esize());
}

template <typename _Tp>
const _Tp* _Matrix<_Tp>::ptr(int row, int col) const
{
    assert(!(static_cast<unsigned>(row) >= static_cast<unsigned>(rows)
        || static_cast<unsigned>(col) >= static_cast<unsigned>(cols)));

    return reinterpret_cast<_Tp *>(data + row * step + col * esize());
}


template <typename _Tp>
_Tp* _Matrix<_Tp>::ptr(int row, int col, int ch)
{
    assert(!(static_cast<unsigned>(row) >= static_cast<unsigned>(rows)
        || static_cast<unsigned>(col) >= static_cast<unsigned>(cols)
        || static_cast<unsigned>(ch) >= static_cast<unsigned>(channels())));

    return reinterpret_cast<_Tp *>(data + row * step + col * esize() + ch * sizeof(_Tp));
}

template <typename _Tp>
const _Tp* _Matrix<_Tp>::ptr(int row, int col, int ch) const
{
    assert(!(static_cast<unsigned>(row) >= static_cast<unsigned>(rows)
        || static_cast<unsigned>(col) >= static_cast<unsigned>(cols)
        || static_cast<unsigned>(ch) >= static_cast<unsigned>(channels())));

    return reinterpret_cast<_Tp *>(data + row * step + col * esize() + ch * sizeof(_Tp));
}

template <typename _Tp>
template<typename _T2> _T2* _Matrix<_Tp>::ptr(int row)
{
    assert(!(static_cast<unsigned>(row) >= static_cast<unsigned>(rows)));

    return reinterpret_cast<_T2 *>(data + row * step);
}

template <typename _Tp>
template<typename _T2> const _T2* _Matrix<_Tp>::ptr(int row) const
{
    assert(!(static_cast<unsigned>(row) >= static_cast<unsigned>(rows)));

    return reinterpret_cast<const _T2 *>(data + row * step);
}

template<typename _Tp>
template<typename _T2>
_T2* _Matrix<_Tp>::ptr(int row, int col)
{
    assert(!(static_cast<unsigned>(row) >= static_cast<unsigned>(rows)
        || static_cast<unsigned>(col) >= static_cast<unsigned>(cols)));

    return reinterpret_cast<_T2 *>(data + row * step + col * esize());
}

template <typename _Tp>
template<typename _T2> const _T2* _Matrix<_Tp>::ptr(int row, int col) const
{
    assert(!(static_cast<unsigned>(row) >= static_cast<unsigned>(rows)
        || static_cast<unsigned>(col) >= static_cast<unsigned>(cols)));

    return reinterpret_cast<const _T2 *>(data + row * step + col * esize());
}

template <typename _Tp>
template<typename _T2> _T2* _Matrix<_Tp>::ptr(int row, int col, int ch)
{
    assert(!(static_cast<unsigned>(row) >= static_cast<unsigned>(rows)
        || static_cast<unsigned>(col) >= static_cast<unsigned>(cols)
        || static_cast<unsigned>(ch) >= static_cast<unsigned>(channels())));

    return reinterpret_cast<_T2 *>(data + row * step + col * esize() + ch * sizeof(_Tp));
}

template <typename _Tp>
template<typename _T2> const _T2* _Matrix<_Tp>::ptr(int row, int col, int ch) const
{
    assert(!(static_cast<unsigned>(row) >= static_cast<unsigned>(rows)
        || static_cast<unsigned>(col) >= static_cast<unsigned>(cols)
        || static_cast<unsigned>(ch) >= static_cast<unsigned>(channels())));

    return reinterpret_cast<const _T2 *>(data + row * step + col * esize() + ch * sizeof(_Tp));
}

template <typename _Tp>
_Tp& _Matrix<_Tp>::at(int row)
{
    assert(static_cast<unsigned>(row) < static_cast<unsigned>(rows));
    return *reinterpret_cast<_Tp *>(data + row * step);
}

template <typename _Tp>
const _Tp& _Matrix<_Tp>::at(int row) const
{
    assert(static_cast<unsigned>(row) < static_cast<unsigned>(rows));
    return *reinterpret_cast<const _Tp *>(data + row * step);
}

template <typename _Tp>
_Tp& _Matrix<_Tp>::at(int row, int col)
{
    assert(!(static_cast<unsigned>(row) >= static_cast<unsigned>(rows)
        || static_cast<unsigned>(col) >= static_cast<unsigned>(cols)));

    return *reinterpret_cast<_Tp *>(data + row * step + col * esize());
}

template <typename _Tp>
const _Tp& _Matrix<_Tp>::at(int row, int col) const
{
    assert(!(static_cast<unsigned>(row) >= static_cast<unsigned>(rows)
        || static_cast<unsigned>(col) >= static_cast<unsigned>(cols)));

    return *reinterpret_cast<const _Tp *>(data + row * step + col * esize());
}

template <typename _Tp>
_Tp& _Matrix<_Tp>::at(int row, int col, int ch)
{
    assert(!(static_cast<unsigned>(row) >= static_cast<unsigned>(rows)
        || static_cast<unsigned>(col) >= static_cast<unsigned>(cols)
        || static_cast<unsigned>(ch) >= static_cast<unsigned>(channels())));

    return *reinterpret_cast<_Tp *>(data + row * step + col * esize() + ch * sizeof(_Tp));
}

template <typename _Tp>
const _Tp& _Matrix<_Tp>::at(int row, int col, int ch) const
{
    assert(!(static_cast<unsigned>(row) >= static_cast<unsigned>(rows)
        || static_cast<unsigned>(col) >= static_cast<unsigned>(cols)
        || static_cast<unsigned>(ch) >= static_cast<unsigned>(channels())));

    return *reinterpret_cast<const _Tp *>(data + row * step + col * esize() + ch * sizeof(_Tp));
}

template <typename _Tp>
template <typename _T2>
_T2& _Matrix<_Tp>::at(int row)
{
    assert(static_cast<unsigned>(row) < static_cast<unsigned>(rows));
    return *reinterpret_cast<_T2 *>(data + row * step);
}

template <typename _Tp>
template <typename _T2>
const _T2& _Matrix<_Tp>::at(int row) const
{
    assert(static_cast<unsigned>(row) < static_cast<unsigned>(rows));
    return *reinterpret_cast<const _T2 *>(data + row * step);
}

template <typename _Tp>
template <typename _T2>
inline _T2& _Matrix<_Tp>::at(int row, int col)
{
    assert(!(static_cast<unsigned>(row) >= static_cast<unsigned>(rows)
        || static_cast<unsigned>(col) >= static_cast<unsigned>(cols)));

    return *reinterpret_cast<_T2 *>(data + row * step + col * esize());
}

template <typename _Tp>
template <typename _T2>
const _T2& _Matrix<_Tp>::at(int row, int col) const
{
    assert(!(static_cast<unsigned>(row) >= static_cast<unsigned>(rows)
        || static_cast<unsigned>(col) >= static_cast<unsigned>(cols)));

    return *reinterpret_cast<const _T2 *>(data + row * step + col * esize());
}

template <typename _Tp>
template <typename _T2>
_T2& _Matrix<_Tp>::at(int row, int col, int ch)
{
    assert(!(static_cast<unsigned>(row) >= static_cast<unsigned>(rows)
        || static_cast<unsigned>(col) >= static_cast<unsigned>(cols)
        || static_cast<unsigned>(ch) >= static_cast<unsigned>(channels())));

    return *reinterpret_cast<_T2 *>(data + row * step + col * esize() + ch * sizeof(_Tp));
}

template <typename _Tp>
template <typename _T2>
const _T2& _Matrix<_Tp>::at(int row, int col, int ch) const
{
    assert(!(static_cast<unsigned>(row) >= static_cast<unsigned>(rows)
        || static_cast<unsigned>(col) >= static_cast<unsigned>(cols)
        || static_cast<unsigned>(ch) >= static_cast<unsigned>(channels())));

    return *reinterpret_cast<const _T2 *>(data + row * step + col * esize() + ch * sizeof(_Tp));
}

template <typename _Tp>
_MatrixIterator<_Tp> _Matrix<_Tp>::begin()
{
    assert(esize() == sizeof(_Tp));
    return _MatrixIterator<_Tp>(this);
}

template <typename _Tp>
_MatrixConstIterator<_Tp> _Matrix<_Tp>::begin() const
{
    assert(esize() == sizeof(_Tp));
    return _MatrixConstIterator<_Tp>(this);
}

template <typename _Tp>
template <typename _T2>
_MatrixIterator<_T2> _Matrix<_Tp>::begin()
{
    assert(esize() == sizeof(_T2));
    return _MatrixIterator<_T2>(reinterpret_cast<_Matrix<_T2>*>(this));
}

template <typename _Tp>
template <typename _T2>
_MatrixConstIterator<_T2> _Matrix<_Tp>::begin() const
{
    assert(esize() == sizeof(_T2));
    return _MatrixConstIterator<_T2>(reinterpret_cast<const _Matrix<_T2>*>(this));
}

template <typename _Tp>
_MatrixIterator<_Tp> _Matrix<_Tp>::end()
{
    assert(esize() == sizeof(_Tp));
    auto r = _MatrixIterator<_Tp>(this);
    r += total();
    return r;
}

template <typename _Tp>
_MatrixConstIterator<_Tp> _Matrix<_Tp>::end() const
{
    assert(esize() == sizeof(_Tp));
    auto r = _MatrixConstIterator<_Tp>(this);
    r += total();
    return r;
}

template <typename _Tp>
template <typename _T2>
_MatrixIterator<_T2> _Matrix<_Tp>::end()
{
    assert(esize() == sizeof(_T2));
    auto r = _MatrixIterator<_T2>(reinterpret_cast<_Matrix<_T2>*>(this));
    r += total();
    return r;
}

template <typename _Tp>
template <typename _T2>
_MatrixConstIterator<_T2> _Matrix<_Tp>::end() const
{
    assert(esize() == sizeof(_T2));
    auto r = _MatrixConstIterator<_T2>(reinterpret_cast<const _Matrix<_T2>*>(this));
    r += total();
    return r;
}

template <typename _Tp>
_Matrix<_Tp> _Matrix<_Tp>::inv()
{
    _Matrix<_Tp> m(cols, rows);
    // do something..
    return m;
}

template <typename _Tp>
_Matrix<_Tp> _Matrix<_Tp>::t()
{
    _Matrix<_Tp> m(cols, rows, channels());

    for (auto i = 0; i < m.rows; ++i) {
        for (auto j = 0; j < m.cols; ++j) {
            for (auto k = 0; k < channels(); ++k) {
                m.at<_Tp>(i, j, k) = at<_Tp>(j, i, k);
            }
        }
    }
    return m;
}

template <typename _Tp>
_Matrix<_Tp> _Matrix<_Tp>::dot(_Matrix<_Tp> &m)
{
    if (rows != m.rows || cols != m.cols || channels() != m.channels())
        _log_("rows != m.rows || cols != m.cols || || chs != m.chs");

    _Matrix<_Tp> temp(m.rows, m.cols, m.channels());

    for (size_t i = 0; datastart + i < dataend; ++i) {
        temp.data[i] = data[i] * m.data[i];
    }

    return temp;
}

template <typename _Tp>
_Matrix<_Tp> _Matrix<_Tp>::cross(_Matrix<_Tp> &m)
{
    if (rows != 1 || cols != 3 || m.rows != 1 || m.cols != 3 || channels() != 0)
        _log_("rows != 1 || cols != 3 || m.rows != 1 || m.cols != 3 || chs != 0");

    _Matrix<_Tp> temp(1, 3);

    temp[0][0] = data[1] * m.data[2] - data[2] * m.data[1];
    temp[0][1] = data[2] * m.data[0] - data[0] * m.data[2];
    temp[0][2] = data[0] * m.data[1] - data[1] * m.data[0];

    return temp;
}

template<typename _Tp>
void _Matrix<_Tp>::swap(int32_t i0, int32_t j0, int32_t i1, int32_t j1) {
    for (uint8_t k = 0; k < channels(); ++k) {
        _Tp temp = ptr(i0, j0)[k];
        ptr(i0, j0)[k] = ptr(i1, j1)[k];
        ptr(i1, j1)[k] = temp;
    }
}

template <typename _Tp>
_Tp* _Matrix<_Tp>::operator[](size_t n)
{
    assert(!(static_cast<unsigned>(n) >= static_cast<unsigned>(rows)));

    return reinterpret_cast<_Tp *>(data + n * step);
}

template <typename _Tp>
const _Tp* _Matrix<_Tp>::operator[](size_t n) const
{
    assert(!(static_cast<unsigned>(n) >= static_cast<unsigned>(rows)));

    return reinterpret_cast<const _Tp*>(data + n * step);
}


template <typename _Tp>
_Matrix<_Tp> operator+(const _Matrix<_Tp>& m1, const _Matrix<_Tp>& m2)
{
    assert(m1.size() == m2.size());
    assert(m1.channels() == m2.channels());

    auto rm = z::_Matrix<_Tp>(m1.size(), m1.channels());

    auto _begin_1 = m1.begin();
    auto _begin_2 = m2.begin();
    auto _begin_3 = rm.begin();
    for (; _begin_1 != m1.end(); ++_begin_1, ++_begin_2, ++_begin_3) {
        *_begin_3 = saturate_cast<_Tp>(*_begin_1 + *_begin_2);
    }

    return rm;
}

template <typename _Tp>
_Matrix<_Tp> operator+(const _Matrix<_Tp>& m, const Scalar& delta)
{
    assert(m.channels() <= 4);

    auto rm = z::_Matrix<_Tp>(m.size(), m.channels());

    for (auto i = 0; i < m.rows; ++i) {
        for (auto j = 0; j < m.cols; ++j) {
            for (auto k = 0; k < m.channels(); ++k) {
                auto&& _value = rm.at(i, j, k);
                _value = saturate_cast<_Tp>(_value + delta[k]);
            }
        }
    }
    return rm;
}

template <typename _Tp>
_Matrix<_Tp> operator+(const Scalar& delta, const _Matrix<_Tp>& m)
{
    return m + delta;
}

template <class _Tp>
_Matrix<_Tp> operator+=(_Matrix<_Tp>& m1, const _Matrix<_Tp>& m2)
{
    assert(m1.rows == m2.rows && m1.cols == m2.cols);

    auto _begin_1 = m1.begin();
    auto _begin_2 = m2.begin();

    for(; _begin_1 != m1.end(); ++_begin_1, ++_begin_2) {
        *_begin_1 = saturate_cast<_Tp>(*_begin_1 + *_begin_2);
    }
    return m1;
}

template <class _Tp>
_Matrix<_Tp> operator+=(_Matrix<_Tp>& m, const Scalar& delta)
{
    assert(m.channels() <= 4);

    for(auto i = 0; i < m.rows; ++i) {
        for(auto j = 0; j < m.cols; ++j) {
            for(auto k = 0; k < m.channels(); ++k) {
                auto&& _value = m.at(i, j, k);
                _value = saturate_cast<_Tp>(_value + delta[k]);
            }
        }
    }
    return m;
}

template <typename _Tp>
_Matrix<_Tp> operator-(const _Matrix<_Tp>& m1, const _Matrix<_Tp>& m2)
{
    assert(m1.size() == m2.size());
    assert(m1.channels() == m2.channels());

    auto rm = z::_Matrix<_Tp>(m1.size(), m1.channels());

    auto _begin_1 = m1.begin();
    auto _begin_2 = m2.begin();
    auto _begin_3 = rm.begin();
    for(; _begin_1 != m1.end(); ++_begin_1, ++_begin_2, ++_begin_3) {
        *_begin_3 = saturate_cast<_Tp>(*_begin_1 - *_begin_2);
    }

    return rm;
}

template <typename _Tp>
_Matrix<_Tp> operator-(const _Matrix<_Tp>& m, const Scalar& delta)
{
    return m + (-delta);
}

template <typename _Tp>
_Matrix<_Tp> operator-(const Scalar& delta, const _Matrix<_Tp>& m)
{
    return (-1 * m) + delta;
}

template <typename _Tp>
_Matrix<_Tp> operator-=(_Matrix<_Tp>& m1, const _Matrix<_Tp>& m2)
{
    assert(m1.size() == m2.size());
    assert(m1.channels() == m2.channels())

    auto _begin_1 = m1.begin();
    auto _begin_2 = m2.begin();

    for (; _begin_1 != m1.end(); ++_begin_1, ++_begin_2) {
        (*_begin_1) = saturate_cast<_Tp>((*_begin_1) - (*_begin_2));
    }
    return m1;
}

template <class _Tp>
_Matrix<_Tp> operator-=(_Matrix<_Tp>& m, const Scalar& delta)
{
    assert(m.channels() <= 4);

    for (auto i = 0; i < m.rows; ++i) {
        for (auto j = 0; j < m.cols; ++j) {
            for (auto k = 0; k < m.channels(); ++k) {
                auto&& _value = m.at(i, j, k);
                _value = saturate_cast<_Tp>(_value - delta[k]);
            }
        }
    }
    return m;
}

template <class _Tp>
_Matrix<_Tp> operator*(const _Matrix<_Tp>& m1, const _Matrix<_Tp>& m2)
{
    assert(m1.cols == m2.rows);
    assert(m1.channels() == 1 && m2.channels() == 1);

    auto rm = _Matrix<_Tp>::zeros(m1.rows, m2.cols);
    for(auto i = 0; i < rm.rows; ++i) {
        for(auto j = 0; j < rm.cols; ++j) {
            for(auto k = 0; k < m1.cols; ++k) {
                rm.at(i, j) = saturate_cast<_Tp>(rm.at(i, j) + m1.at(i, k) * m2.at(k, j));
            }
        }
    }
    return rm;
}

template <typename _Tp>
_Matrix<_Tp> operator*(const _Matrix<_Tp>& m, double v)
{
    auto rm = z::_Matrix<_Tp>(m.size(), m.channels());

    auto _begin_1 = m.begin();
    auto _begin_2 = rm.begin();

    for(; _begin_1 != m.end(); ++_begin_1, ++_begin_2) {
        *_begin_2 = saturate_cast<_Tp>((*_begin_1) * v);
    }
    return rm;
}

template <typename _Tp>
_Matrix<_Tp> operator*(double v, const _Matrix<_Tp>& m)
{
    return m * v;
}

template <class _Tp>
_Matrix<_Tp> operator*=(_Matrix<_Tp>& m1, const _Matrix<_Tp>& m2)
{
    m1 = m1 * m2;
    return m1;
}

template <class _Tp>
_Matrix<_Tp> operator*=(_Matrix<_Tp>& m, double v)
{
    for(auto&pixel : m) {
        pixel = saturate_cast<_Tp>(pixel * v);
    }
    return m;
}

template <class _Tp>
_Matrix<_Tp> operator/(const _Matrix<_Tp>& m1, const _Matrix<_Tp>& m2)
{
}

template <class _Tp>
_Matrix<_Tp> operator/(const _Matrix<_Tp>& m, double v)
{
    return m * (1 / v);
}

template <class _Tp>
_Matrix<_Tp> operator/(double v, const _Matrix<_Tp>& m)
{
    return m * (1 / v);
}

template <class _Tp>
_Matrix<_Tp> operator/=(_Matrix<_Tp>& m, double v)
{
    m *= 1 / v;
    return m;
}

template <typename _Tp>
_Matrix<_Tp> operator>(const _Matrix<_Tp>& m, double threshold)
{
    auto rm = z::_Matrix<_Tp>(m.size(), m.channels());

    auto _begin_1 = m.begin();
    auto _begin_2 = rm.begin();

    for(; _begin_1 != m.end(); ++_begin_1, ++_begin_2) {
        *_begin_1 > threshold ? *_begin_2 = std::numeric_limits<_Tp>::max() : *_begin_2 = std::numeric_limits<_Tp>::min();
    }
    return rm;
}

template <typename _Tp>
_Matrix<_Tp> operator>(double threshold, const _Matrix<_Tp>& m)
{
    return m < threshold;
}

template <typename _Tp>
_Matrix<_Tp> operator>=(_Matrix<_Tp>& m, double threshold)
{
    for (auto&pixel : m) {
        pixel >= threshold ? pixel = std::numeric_limits<_Tp>::max() : pixel = std::numeric_limits<_Tp>::min();
    }
    return m;
}

template <typename _Tp>
_Matrix<_Tp> operator<(const _Matrix<_Tp>& m, double threshold)
{
    auto rm = z::_Matrix<_Tp>(m.size(), m.channels());
    auto ele_num = m.cols * m.channels();

    for (auto i = 0; i < m.rows; ++i) {
        auto ptr = rm.template ptr<_Tp>(i);
        auto mptr = m.template ptr<_Tp>(i);
        for (auto j = 0; j < ele_num; ++j) {
            mptr[j] < threshold ? ptr[j] = std::numeric_limits<_Tp>::max() : ptr[j] = std::numeric_limits<_Tp>::min();
        }
    }
    return rm;
}

template <typename _Tp>
_Matrix<_Tp> operator < (double threshold, const _Matrix<_Tp>& m)
{
    return m > threshold;
}

template <typename _Tp>
_Matrix<_Tp> operator<=(_Matrix<_Tp>& m, double threshold)
{
    for (auto&pixel : m) {
        pixel <= threshold ? pixel = std::numeric_limits<_Tp>::max() : pixel = std::numeric_limits<_Tp>::min();
    }
    return m;
}

template <typename _Tp>
_Matrix<_Tp> operator==(const _Matrix<_Tp>& m1, const _Matrix<_Tp>& m2)
{
    assert(m1.size() == m2.size());
    assert(m1.channels() == m2.channels());

    auto rm = z::_Matrix<_Tp>(m1.size(), m1.channels());
    
    auto _begin_1 = m1.begin();
    auto _begin_2 = m2.begin();
    auto _begin_3 = rm.begin();

    for(; _begin_1 != m1.end(); ++_begin_1, ++_begin_2) {
        *_begin_1 == *_begin_2 ? *_begin_3 = td::numeric_limits<_Tp>::max() : *_begin_3 = std::numeric_limits<_Tp>::min();
    }
    return rm;
}

template <typename _Tp>
_Matrix<_Tp> operator==(const _Matrix<_Tp>& m, double val)
{
    auto rm = z::_Matrix<_Tp>(m.size(), m.channels());

    auto _begin_1 = m.begin();
    auto _begin_2 = rm.begin();

    for(; _begin_1 != m.end(); ++_begin_1, ++_begin_2) {
        *_begin_1 == val ? *_begin_2 = std::numeric_limits<_Tp>::max() : *_begin_2 = std::numeric_limits<_Tp>::min();
    }

    return rm;
}

template <typename _Tp>
_Matrix<_Tp> operator==(double val, const _Matrix<_Tp>& m)
{
    return m == val;
}

template <typename _Tp>
_Matrix<_Tp> operator!=(const _Matrix<_Tp>& m1, const _Matrix<_Tp>& m2)
{
    assert(m1.size() == m2.size());
    assert(m1.channels() == m2.channels());

    auto rm = z::_Matrix<_Tp>(m1.size(), m1.channels());

    auto _begin_1 = m1.begin();
    auto _begin_2 = m2.begin();
    auto _begin_3 = rm.begin();

    for (; _begin_1 != m1.end(); ++_begin_1, ++_begin_2) {
        *_begin_1 != *_begin_2 ? *_begin_3 = td::numeric_limits<_Tp>::max() : *_begin_3 = std::numeric_limits<_Tp>::min();
    }
    return rm;
}

template <typename _Tp>
_Matrix<_Tp> operator!=(const _Matrix<_Tp>& m, double val)
{
    auto rm = z::_Matrix<_Tp>(m.size(), m.channels());

    auto _begin_1 = m.begin();
    auto _begin_2 = rm.begin();

    for (; _begin_1 != m.end(); ++_begin_1, ++_begin_2) {
        *_begin_1 != val ? *_begin_2 = std::numeric_limits<_Tp>::max() : *_begin_2 = std::numeric_limits<_Tp>::min();
    }

    return rm;
}

template <typename _Tp>
_Matrix<_Tp> operator!=(double val, const _Matrix<_Tp>& m)
{
    return m != val;
}


template <typename _Tp>
std::ostream &operator<<(std::ostream & os, const _Matrix<_Tp> &item)
{
    for (auto i = 0; i < item.rows; ++i) {
        for (auto j = 0; j < item.cols; ++j) {
            os << "[";
            for (auto k = 0; k < item.channels(); ++k) {
                sizeof(_Tp) == 1 ? os << static_cast<int>(item.ptr(i, j)[k]) : os << item[i][j];

                if (k + 1 != item.channels())
                    os << ", ";
            }
            os << "]";
            if (j + 1 != item.cols)
                os << ", ";
        }
        os << ';' << std::endl;
    }
    return os;
}

/////////////////////////////////////////_MatrixConstIterator////////////////////////////////////////////
template <typename _Tp>
_MatrixConstIterator<_Tp>::_MatrixConstIterator(const _Matrix<_Tp>* m) :
    m_(m), esize_(m->esize())
{
    if (m_ && m_->isContinuous()) {
        start_ = m->template ptr<uint8_t>();
        end_ = start_ + m_->total() * esize_;
    }

    seek(0, false);
}

template <typename _Tp>
_MatrixConstIterator<_Tp>::_MatrixConstIterator(const _MatrixConstIterator<_Tp>& it)
    :m_(it.m_), esize_(it.esize_), ptr_(it.ptr_), start_(it.start_), end_(it.end_)
{}

template <typename _Tp>
_MatrixConstIterator<_Tp>& _MatrixConstIterator<_Tp>::operator=(const _MatrixConstIterator<_Tp>& it)
{
    m_ = it.m_;
    esize_ = it.esize_;
    ptr_ = it.ptr_;
    start_ = it.start_;
    end_ = it.end_;

    return *this;
}

template <typename _Tp>
const _Tp& _MatrixConstIterator<_Tp>::operator*() const
{
    return *reinterpret_cast<const _Tp *>(ptr_);
}

template <typename _Tp>
const _Tp& _MatrixConstIterator<_Tp>::operator[](ptrdiff_t i) const
{
    return *reinterpret_cast<const _Tp *>(*this + i);
}

template <typename _Tp>
_MatrixConstIterator<_Tp>& _MatrixConstIterator<_Tp>::operator+=(ptrdiff_t ofs)
{
    if (!m_ || !ofs) return *this;
    ptrdiff_t ofsb = ofs * esize_;
    ptr_ += ofsb;

    if (ptr_ < start_ || end_ <= ptr_)
    {
        ptr_ -= ofsb;
        seek(ofs, true);
    }
    return *this;
}

template <typename _Tp>
_MatrixConstIterator<_Tp>& _MatrixConstIterator<_Tp>::operator-=(ptrdiff_t ofs)
{
    return *this += -ofs;
}

template <typename _Tp>
_MatrixConstIterator<_Tp>& _MatrixConstIterator<_Tp>::operator--()
{
    if (!!m_ && (ptr_ -= esize_) < start_) {
        ptr_ += esize_;
        seek(-1, true);
    }
    return *this;
}

template <typename _Tp>
_MatrixConstIterator<_Tp>& _MatrixConstIterator<_Tp>::operator--(int)
{
    auto r = *this;
    *this += -1;
    return r;
}

template <typename _Tp>
_MatrixConstIterator<_Tp>& _MatrixConstIterator<_Tp>::operator++()
{
    if (!!m_ && (ptr_ += esize_) >= end_) {
        ptr_ -= esize_;
        seek(1, true);
    }
    return *this;
}

template <typename _Tp>
_MatrixConstIterator<_Tp>& _MatrixConstIterator<_Tp>::operator++(int)
{
    auto r = *this;
    *this += 1;
    return r;
}

template <typename _Tp>
bool _MatrixConstIterator<_Tp>::operator==(const _MatrixConstIterator<_Tp>& it) const
{
    return ptr_ == it.ptr_;
}

template <typename _Tp>
bool _MatrixConstIterator<_Tp>::operator!=(const _MatrixConstIterator<_Tp>& it) const
{
    return ptr_ != it.ptr_;
}

template <typename _Tp>
void _MatrixConstIterator<_Tp>::seek(ptrdiff_t ofs, bool relative)
{
    if (m_->isContinuous()) {
        ptr_ = (relative ? ptr_ : start_) + ofs * esize_;

        if (ptr_ < start_)
            ptr_ = start_;
        else if (ptr_ > end_)
            ptr_ = end_;

        return;
    }

    ptrdiff_t row;
    if (relative) {
        ptrdiff_t ofs0 = ptr_ - m_->template ptr<uint8_t>();
        row = ofs0 / m_->step;
        ofs += row * m_->cols + (ofs0 - row * m_->step) / esize_;
    }
    row = ofs / m_->cols;
    int y1 = std::min(std::max(int(row), 0), m_->rows - 1);
    start_ = m_->template ptr<uint8_t>(y1);
    end_ = start_ + m_->cols * esize_;
    ptr_ = row < 0 ? start_ :
        (row >= m_->rows ? end_ : start_ + (ofs - row * m_->cols) * esize_);
}

template <typename _Tp>
_MatrixIterator<_Tp>::_MatrixIterator(_Matrix<_Tp>* _m)
    :_MatrixConstIterator<_Tp>(_m)
{}

template <typename _Tp>
_MatrixIterator<_Tp>::_MatrixIterator(const _MatrixIterator& it)
    : _MatrixConstIterator<_Tp>(it)
{}

template <typename _Tp>
_MatrixIterator<_Tp>& _MatrixIterator<_Tp>::operator=(const _MatrixIterator<_Tp>& it)
{
    _MatrixConstIterator<_Tp>::operator=(it);
    return *this;
}

template <typename _Tp>
_Tp& _MatrixIterator<_Tp>::operator*() const
{
    return *(_Tp *)this->ptr_;
}

template <typename _Tp>
_Tp& _MatrixIterator<_Tp>::operator[](ptrdiff_t i) const
{
    return *(*this + i);
}

template <typename _Tp>
_MatrixIterator<_Tp>& _MatrixIterator<_Tp>::operator+=(ptrdiff_t ofs)
{
    _MatrixConstIterator<_Tp>::operator+=(ofs);
    return *this;
}

template <typename _Tp>
_MatrixIterator<_Tp>& _MatrixIterator<_Tp>::operator-=(ptrdiff_t ofs)
{
    _MatrixConstIterator<_Tp>::operator-=(ofs);
    return *this;
}

template <typename _Tp>
_MatrixIterator<_Tp>& _MatrixIterator<_Tp>::operator--()
{
    _MatrixConstIterator<_Tp>::operator--();
    return *this;
}

template <typename _Tp>
_MatrixIterator<_Tp> _MatrixIterator<_Tp>::operator--(int)
{
    auto r = *this;
    _MatrixConstIterator<_Tp>::operator --();
    return r;
}

template <typename _Tp>
_MatrixIterator<_Tp>& _MatrixIterator<_Tp>::operator++()
{
    _MatrixConstIterator<_Tp>::operator++();
    return *this;
}

template <typename _Tp>
_MatrixIterator<_Tp> _MatrixIterator<_Tp>::operator++(int)
{
    auto r = *this;
    _MatrixConstIterator<_Tp>::operator ++();
    return r;
}
}

#endif // ! _OPERATIONS_HPP
