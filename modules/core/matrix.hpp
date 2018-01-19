#ifndef ALCHEMY_CORE_MATRIX_HPP
#define ALCHEMY_CORE_MATRIX_HPP

#include <cstring>
#include <algorithm>
#include <limits>
#include <iomanip>
#include "util/saturate.h"
#include "util/util.h"

#define ZMATRIX_CONTINUOUS_MASK (1 << 16)
#define ZMATRIX_CH_MASK         (0x000000ff)
#define ZMATRIX_TYPE_MASK       (0x0000ffff)
#define ZMATRIX_DEPTH_MASK      (0x0000ff00)

namespace alchemy {

/////////////////////////////////////////////////////////////////////////////////////////////
template <typename _Tp>
_Matrix<_Tp>::_Matrix(int rows, int cols, int chs)
{
    create(rows, cols, chs);
}

template <typename _Tp>
_Matrix<_Tp>::_Matrix(Size size, int chs)
        : _Matrix(size.height, size.width, chs)
{}

template <typename _Tp>
_Matrix<_Tp>::_Matrix(const MatrixShape& shape)
        : _Matrix(shape.rows, shape.cols, shape.chs)
{}

template <typename _Tp>
template<typename _T2>
_Matrix<_Tp>::_Matrix(int rows, int cols, int chs, const _T2& v)
{
    create(rows, cols, chs);
    fill(v);
}

template <typename _Tp>
template<typename _T2>
_Matrix<_Tp>::_Matrix(Size size, int chs, const _T2& v)
        :_Matrix(size.height, size.width, chs, v)
{}

template <typename _Tp>
template<typename _T2>
_Matrix<_Tp>::_Matrix(const MatrixShape& shape, const _T2& v)
        : _Matrix(shape.rows, shape.cols, shape.chs, v)
{}

template <typename _Tp>
template<typename _T2, typename _T3>
_Matrix<_Tp>::_Matrix(int rows, int cols, int chs, const std::pair<_T2, _T3>& initor)
{
    create(rows, cols, chs);

    int _len = rows * cols * chs;
    for(auto i = 0; i < _len; ++i) {
        reinterpret_cast<_Tp *>(data)[i] = saturate_cast<_Tp>(const_cast<_T3&>(initor.second)(const_cast<_T2&>(initor.first)));
    }
}

template <typename _Tp>
template<typename _T2, typename _T3>
_Matrix<_Tp>::_Matrix(Size size, int chs, const std::pair<_T2, _T3>& initor)
        : _Matrix(size.height, size.width, chs, initor)
{}

template <typename _Tp>
template <typename _T2, typename _T3>
_Matrix<_Tp>::_Matrix(const MatrixShape& shape, const std::pair<_T2, _T3>& initor)
        : _Matrix(shape.rows, shape.cols, shape.chs, initor)
{}

template <typename _Tp>
_Matrix<_Tp>::_Matrix(const _Matrix& m)
{
    *this = m;
}

template <typename _Tp>
template <typename _T2>
_Matrix<_Tp>::_Matrix(const _Matrix<_T2>& m)
{
    flags = (flags & ~ZMATRIX_TYPE_MASK) | DataType<_Tp>::value;
    *this = m;
}

template <typename _Tp>
_Matrix<_Tp>::_Matrix(const _Matrix<_Tp>& m, const Rect& roi)
        :rows(roi.height), cols(roi.width), step(m.step), flags(m.flags),
         datastart(m.datastart), dataend(m.dataend),
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

    size_ = static_cast<size_t>(rows * cols);

    refAdd(refcount, 1);
}

template <typename _Tp>
_Matrix<_Tp>::_Matrix(std::initializer_list<_Tp> list)
{
    if (list.size() == 0) return;

    create(static_cast<int>(list.size()), 1, 1);
    fill(0);

    auto begin = ptr();
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

        release();

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
    if (DataType<_Tp>::value == m.type()) {
        auto mptr = reinterpret_cast<const _Matrix<_Tp> *>(&m);
        _Matrix::operator=(*mptr);
        return *this;
    }

    // : _Matrix<int8_t>(3, 3, 3) = > _Matrix<Vec_<int8_t, 1>>(3, 9, 1);
    if (DataType<_Tp>::depth == m.depth()) {
        *this = m.reshape(DataType<_Tp>::channels);
        return *this;
    }

    // : _Matrix<int8_t>(3, 3, 3) = > _Matrix<float>(3, 3, 3);
    if (shape() != m.shape())
        create(m.shape());

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

    auto data_ptr = ptr();

    if (isContinuous()) {
        for (const auto&e : list) {
            *data_ptr++ = e;
        }
        return *this;
    }

    auto i = 0;
    for (const auto&e : list) {
        data_ptr = ptr(i / (cols * channels()));
        data_ptr[(i++) % (cols * channels())] = e;
    }
    return *this;
}

template <typename _Tp>
_Matrix<_Tp>::~_Matrix()
{
    release();
}

template <typename _Tp>
template <typename _T2>
void _Matrix<_Tp>::fill(const _T2& value)
{
    // 1 row
    const auto _len = cols * channels();
    for (auto i = 0; i < _len; ++i) {
        reinterpret_cast<_Tp *>(data)[i] = saturate_cast<_Tp>(value);
    }

    // other rows
    // roi: cols * esize_ != step
    const auto _step = cols * esize_;
    for (auto i = 0; i < rows; ++i) {
        memcpy(ptr(i), data, _step);
    }
}

template <typename _Tp>
void _Matrix<_Tp>::fill(const Scalar& s)
{
    assert(channels() <= 4);

    // 1 row
    for (auto i = 0; i < cols; ++i) {
        for (auto j = 0; j < channels(); ++j) {
            at(0, i, j) = saturate_cast<_Tp>(s[j]);
        }
    }

    // other rows
    // roi: cols * esize_ != step
    const auto _step = cols * esize_;
    for (auto i = 0; i < rows; ++i) {
        memcpy(ptr(i), data, _step);
    }
}


template <typename _Tp>
template <typename _T2>
void _Matrix<_Tp>::fill(const _Matrix<_T2>& s)
{
    for (auto i = 0; i < rows; ++i) {
        for (auto j = 0; j < cols; ++j) {
            for (auto k = 0; k < channels(); ++k) {
                at(i, j, k) = _Tp(s.clone());
            }
        }
    }
}

template <typename _Tp>
void _Matrix<_Tp>::create(int _rows, int _cols, int _chs)
{
    assert(_rows * _cols * _chs != 0);

    flags = (DataType<_Tp>::depth << 8) | _chs;
    esize_ = sizeof(_Tp) * _chs;

    rows = _rows;
    cols = _cols;
    step = static_cast<int>(_cols * esize_);
    size_ = static_cast<size_t>(_rows * _cols);

    // Free the memary if the reference counter is 1.
    release();

    // Alloc memory.
    datastart = data = reinterpret_cast<uint8_t*>(new _Tp[size_ * _chs]);
    dataend = data + size_ * esize_;

    // Reference counter.
    refcount = new int(1);
}

template <typename _Tp>
void _Matrix<_Tp>::create(Size size, int chs)
{
    create(size.height, size.width, chs);
}

template <typename _Tp>
void _Matrix<_Tp>::create(const MatrixShape& shape)
{
    create(shape.rows, shape.cols, shape.chs);
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
_Matrix<_Tp> _Matrix<_Tp>::reshape(int cn) const
{
    if (cn == 0 || cn == channels())
        return *this;

    auto width = cols * channels();
    assert(width % cn == 0);

    _Matrix<_Tp> res = *this;

    res.cols = width / cn;
    res.flags = (flags & ~ZMATRIX_CH_MASK) | cn;
    res.size_ = static_cast<size_t>(res.rows * res.cols);
    res.esize_ = sizeof(_Tp) * cn;

    return res;
}

template <typename _Tp>
_Matrix<_Tp> _Matrix<_Tp>::reshape(int cn, int _rows) const
{
    assert(isContinuous());

    if (cn == 0 || empty() || (cn == channels() && _rows == rows))
        return *this;

    _Matrix<_Tp> _r = *this;
    auto _total = _r.total() * channels();

    assert(_total % cn == 0);
    _r.flags = (flags & ~ZMATRIX_CH_MASK) | cn;
    _r.size_ = _total / cn;

    assert(_r.total() % _rows == 0);
    _r.rows = _rows;
    _r.cols = _r.size_ / _rows;
    _r.esize_ = sizeof(_Tp) * cn;
    _r.step = _r.cols * _r.esize_;

    return _r;
}

template <typename _Tp>
_Matrix<_Tp> _Matrix<_Tp>::zeros(int _rows, int _cols, int _chs)
{
    return _Matrix<_Tp>(_rows, _cols, _chs, 0);
}

template <typename _Tp>
_Matrix<_Tp> _Matrix<_Tp>::zeros(Size size, int _chs)
{
    return  _Matrix<_Tp>(size.height, size.width, _chs, 0);
}

template <typename _Tp>
_Matrix<_Tp> _Matrix<_Tp>::zeros(const MatrixShape& shape)
{
    return  _Matrix<_Tp>(shape.rows, shape.cols, shape.chs, 0);
}

template <typename _Tp>
_Matrix<_Tp> _Matrix<_Tp>::ones(int _rows, int _cols, int _chs)
{
    return _Matrix<_Tp>(_rows, _cols, _chs, 1);
}

template <typename _Tp>
_Matrix<_Tp> _Matrix<_Tp>::ones(Size size, int _chs)
{
    return _Matrix<_Tp>(size.height, size.width, _chs, 1);
}

template <typename _Tp>
_Matrix<_Tp> _Matrix<_Tp>::ones(const MatrixShape& shape)
{
    return _Matrix<_Tp>(shape.rows, shape.cols, shape.chs, 1);
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
_Matrix<_Tp> _Matrix<_Tp>::eye(const MatrixShape& shape)
{
    return _Matrix<_Tp>::eye(shape.rows, shape.cols, shape.chs);
}

template <typename _Tp>
void _Matrix<_Tp>::copyTo(_Matrix<_Tp> & outputMatrix) const
{
    outputMatrix.create(rows, cols, channels());
    if (isContinuous() && outputMatrix.isContinuous()) {
        memcpy(outputMatrix.data, data, size_ * esize_);
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

#if defined(USE_OPENCV)
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
    assert(static_cast<unsigned>(row) < static_cast<unsigned>(rows));

    return reinterpret_cast<_T2 *>(data + row * step);
}

template <typename _Tp>
template<typename _T2> const _T2* _Matrix<_Tp>::ptr(int row) const
{
    assert(static_cast<unsigned>(row) < static_cast<unsigned>(rows));

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
_Tp& _Matrix<_Tp>::operator()(int row)
{
    assert(static_cast<unsigned>(row) < static_cast<unsigned>(rows));
    return *reinterpret_cast<_Tp *>(data + row * step);
}

template <typename _Tp>
const _Tp& _Matrix<_Tp>::operator()(int row) const
{
    assert(static_cast<unsigned>(row) < static_cast<unsigned>(rows));
    return *reinterpret_cast<const _Tp *>(data + row * step);
}

template <typename _Tp>
_Tp& _Matrix<_Tp>::operator()(int row, int col)
{
    assert(!(static_cast<unsigned>(row) >= static_cast<unsigned>(rows)
             || static_cast<unsigned>(col) >= static_cast<unsigned>(cols)));

    return *reinterpret_cast<_Tp *>(data + row * step + col * esize());
}

template <typename _Tp>
const _Tp& _Matrix<_Tp>::operator()(int row, int col) const
{
    assert(!(static_cast<unsigned>(row) >= static_cast<unsigned>(rows)
             || static_cast<unsigned>(col) >= static_cast<unsigned>(cols)));

    return *reinterpret_cast<const _Tp *>(data + row * step + col * esize());
}

template <typename _Tp>
_Tp& _Matrix<_Tp>::operator()(int row, int col, int ch)
{
    assert(!(static_cast<unsigned>(row) >= static_cast<unsigned>(rows)
             || static_cast<unsigned>(col) >= static_cast<unsigned>(cols)
             || static_cast<unsigned>(ch) >= static_cast<unsigned>(channels())));

    return *reinterpret_cast<_Tp *>(data + row * step + col * esize() + ch * sizeof(_Tp));
}

template <typename _Tp>
const _Tp& _Matrix<_Tp>::operator()(int row, int col, int ch) const
{
    assert(!(static_cast<unsigned>(row) >= static_cast<unsigned>(rows)
             || static_cast<unsigned>(col) >= static_cast<unsigned>(cols)
             || static_cast<unsigned>(ch) >= static_cast<unsigned>(channels())));

    return *reinterpret_cast<const _Tp *>(data + row * step + col * esize() + ch * sizeof(_Tp));
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
Scalar _Matrix<_Tp>::trace() const {

    Scalar _r;

    for (auto i = 0; i < rows && i < cols; ++i) {
        for (auto k = 0; k < channels(); ++k) {
            _r[k] += at(i, i, k);
        }
    }
    return _r;
}

//template <typename _Tp>
//_Matrix<_Tp> _Matrix<_Tp>::inv()
//{
//	_Matrix<_Tp> m(cols, rows);
//	// do something..
//	return m;
//}

template <typename _Tp>
_Matrix<_Tp> _Matrix<_Tp>::mul(const _Matrix<_Tp> &m, double scale) const
{
    assert(shape() == m.shape());

    _Matrix<_Tp> _r(shape());
    traverse(*this, m, _r, 1, [scale](auto&& _in_1, auto&& _in_2, auto&& _out){
        *_out = saturate_cast<_Tp>(((*_in_1) * (*_in_2)) * scale);
    });

    return _r;
}

template <typename _Tp>
_Matrix<_Matrix<_Tp>> _Matrix<_Tp>::mul(const _Matrix<_Matrix<_Tp>>& m, double scale) const
{
    assert(shape() == m.shape());

    _Matrix<_Matrix<_Tp>> _r(shape());
    traverse(*this, m, _r, 1, [scale](auto&& _in_1, auto&& _in_2, auto&& _out) {
        *_out = ((*_in_1) * (*_in_2)) * scale;
    });

    return _r;
}

template <typename _Tp>
double _Matrix<_Tp>::dot(const _Matrix<_Tp> &m)
{
    assert(shape() == m.shape());

    auto _r = 0.0;

    if(isContinuous() && m.isContinuous()){
        for (size_t i = 0; datastart + i < dataend; ++i) {
            _r += data[i] * m.data[i];
        }
    }
    else {
        auto begin_1 = begin<_Tp>(), end_1 = end<_Tp>();
        auto begin_2 = m.begin<_Tp>();

        for(; begin_1 != end_1; ++begin_1, ++begin_2) {
            _r += (*begin_1) * (*begin_2);
        }
    }

    return _r;
}

template <typename _Tp>
_Matrix<_Tp> _Matrix<_Tp>::cross(_Matrix<_Tp> &m)
{
    assert(size() == m.size() == 3);

    _Matrix<_Tp> temp(1, 3);

    temp[0][0] = data[1] * m.data[2] - data[2] * m.data[1];
    temp[0][1] = data[2] * m.data[0] - data[0] * m.data[2];
    temp[0][2] = data[0] * m.data[1] - data[1] * m.data[0];

    return temp;
}

template <typename _Tp>
_Tp* _Matrix<_Tp>::operator[](size_t n)
{
    assert(static_cast<unsigned>(n) < static_cast<unsigned>(rows));

    return reinterpret_cast<_Tp *>(data + n * step);
}

template <typename _Tp>
const _Tp* _Matrix<_Tp>::operator[](size_t n) const
{
    assert(static_cast<unsigned>(n) < static_cast<unsigned>(rows));

    return reinterpret_cast<const _Tp*>(data + n * step);
}

template <typename _Tp>
template<typename _T2, class Func>
void _Matrix<_Tp>::forEach(const Func &callback)
{
    assert(esize_ == sizeof(_T2));

    for(auto i = 0; i < cols; ++i) {
        auto _ptr = ptr<_T2>(i);
        for(auto j = 0; j < rows; ++j) {
            const int pos[]{cols, rows};
            callback(_ptr[i], pos);
        }
    }
};

template <typename _Tp>
template<class Func>
void _Matrix<_Tp>::forEach(const Func& callback)
{
    assert(esize_ == sizeof(_Tp));

    for(auto i = 0; i < cols; ++i) {
        auto _ptr = ptr(i);
        for(auto j = 0; j < rows; ++j) {
            const int pos[]{cols, rows};
            callback(_ptr[i], pos);
        }
    }
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
int _Matrix<_Tp>::refAdd(int *addr, int delta)
{
    const auto temp = *addr;
    *addr += delta;
    return temp;
}

template <typename _Tp>
void _Matrix<_Tp>::release()
{
    if (refcount && refAdd(refcount, -1) == 1) {
        delete[] reinterpret_cast<_Tp *>(data);
        data = datastart = dataend = nullptr;

        delete refcount;
        refcount = nullptr;
    }
}

///////////////////////////////////////// _Matrix Operators ////////////////////////////////////////////
template <typename _Tp>
_Matrix<_Tp> operator+(const _Matrix<_Tp>& m1, const _Matrix<_Tp>& m2)
{
    _Matrix<_Tp> _r(m1.shape());
    traverse(m1, m2, _r, 1, [](auto&& _in_1, auto&& _in_2, auto&& _out) {
        *_out = saturate_cast<_Tp>(*_in_1 + *_in_2);
    });

    return _r;
}

template <typename _Tp>
_Matrix<_Tp> operator+(const _Matrix<_Tp>& m, const Scalar& delta)
{
    assert(m.channels() <= 4);

    _Matrix<_Tp> _r(m.shape());
    traverse(m, _r, m.channels(), [&](auto&& _in, auto&& _out) {
        for(auto i = 0; i < m.channels(); ++i) {
            _out[i] = saturate_cast<_Tp>(_in[i] + delta[i]);
        }
    });

    return _r;
}

template <typename _Tp>
_Matrix<_Tp> operator+(const Scalar& delta, const _Matrix<_Tp>& m)
{
    return m + delta;
}

template <class _Tp>
_Matrix<_Tp> operator+(const _Matrix<_Tp> &m, double delta)
{
    _Matrix<_Tp> _r(m.shape());
    traverse(m, _r, 1, [=](auto&& _in, auto&& _out) {
        *_out = saturate_cast<_Tp>((*_in) + delta);
    });

    return _r;
}
template <class _Tp>
_Matrix<_Tp> operator+(double delta, const _Matrix<_Tp> &m)
{
    return m + delta;
}

template <class _Tp>
_Matrix<_Tp> operator+=(_Matrix<_Tp>& m1, const _Matrix<_Tp>& m2)
{
    traverse(m2, m1, 1, [](auto&&_in, auto&& _in_out) {
        (*_in_out) = saturate_cast<_Tp>((*_in) + (*_in_out));
    });

    return m1;
}

template <class _Tp>
_Matrix<_Tp> operator+=(_Matrix<_Tp>& m, const Scalar& delta)
{
    assert(m.channels() <= 4);

    traverse(m, m.channels(), [&](auto&&_in_out) {
        for(auto i = 0; i < m.channels(); ++i) {
            _in_out[i] = saturate_cast<_Tp>(_in_out[i] + delta[i]);
        }
    });

    return m;
}

template <class _Tp>
_Matrix<_Tp> operator+=(_Matrix<_Tp> &m, double delta)
{
    traverse(m, 1, [=](auto&&_in_out) {
        *_in_out = saturate_cast<_Tp>((*_in_out) + delta);
    });

    return m;
}

template <typename _Tp>
_Matrix<_Tp> operator-(const _Matrix<_Tp>& m1, const _Matrix<_Tp>& m2)
{
    _Matrix<_Tp> _r(m1.shape());
    traverse(m1, m2, _r, 1, [](auto&& _in_1, auto&& _in_2, auto&& _out) {
        *_out = saturate_cast<_Tp>(*_in_1 - *_in_2);
    });

    return _r;
}

template <typename _Tp>
_Matrix<_Tp> operator-(const _Matrix<_Tp>& m, const Scalar& delta)
{
    return m + (-delta);
}

template <typename _Tp>
_Matrix<_Tp> operator-(const Scalar& delta, const _Matrix<_Tp>& m)
{
    assert(m.channels() <= 4);

    _Matrix<_Tp> _r(m.shape());
    traverse(m, _r, m.channels(), [&](auto&&_in, auto&& _out) {
        for(auto i = 0; i < m.channels(); ++i) {
            _out[i] = saturate_cast<_Tp>(delta[i] - _in[i]);
        }
    });

    return _r;
}

template <class _Tp>
_Matrix<_Tp> operator-(const _Matrix<_Tp> &m, double delta)
{
    return m + (-delta);
}

template <class _Tp>
_Matrix<_Tp> operator-(double delta, const _Matrix<_Tp> &m)
{
    _Matrix<_Tp> _r(m.shape());
    traverse(m, _r, 1, [=](auto&& _in, auto&& _out) {
        *_out = saturate_cast<_Tp>(delta - (*_in));
    });

    return _r;
}

template <typename _Tp>
_Matrix<_Tp> operator-=(_Matrix<_Tp>& m1, const _Matrix<_Tp>& m2)
{
    traverse(m2, m1, 1, [](auto&& _val_2, auto&& _val_1) {
        (*_val_1) = saturate_cast<_Tp>((*_val_1) - (*_val_2));
    });

    return m1;
}

template <class _Tp>
_Matrix<_Tp> operator-=(_Matrix<_Tp>& m, const Scalar& delta)
{
    assert(m.channels() <= 4);

    traverse(m, m.channels(), [&](auto&&_in_out) {
        for(auto i = 0; i < m.channels(); ++i) {
            _in_out[i] = saturate_cast<_Tp>(_in_out[i] - delta[i]);
        }
    });

    return m;
}

template <class _Tp>
_Matrix<_Tp> operator-=(_Matrix<_Tp> &m, double delta)
{
    traverse(m, 1, [=](auto&& _val) {
        *_val = saturate_cast<_Tp>(*_val - delta);
    });

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

template <class _Tp>
_Matrix<_Matrix<_Tp>> operator*(const _Matrix<_Tp> &m1, const _Matrix<_Matrix<_Tp>> &m2)
{
    assert(m1.cols == m2.rows);
    assert(m1.channels() == 1 && m2.channels() == 1);

    _Matrix<_Matrix<_Tp>> rm(m1.rows, m2.cols, 1, _Matrix<_Tp>(m2.at(0).shape(), 0));
    for (auto i = 0; i < rm.rows; ++i) {
        for (auto j = 0; j < rm.cols; ++j) {
            for (auto k = 0; k < m1.cols; ++k) {
                rm.at(i, j) = rm.at(i, j) + m1.at(i, k) * m2.at(k, j);
            }
        }
    }
    return rm;
}
template <class _Tp>
_Matrix<_Matrix<_Tp>> operator*(const _Matrix<_Matrix<_Tp>> &m1, const _Matrix<_Tp> &m2)
{
    assert(m1.cols == m2.rows);
    assert(m1.channels() == 1 && m2.channels() == 1);

    _Matrix<_Matrix<_Tp>> rm(m1.rows, m2.cols, 1, _Matrix<_Tp>(m1.at(0).shape(), 0));
    for (auto i = 0; i < rm.rows; ++i) {
        for (auto j = 0; j < rm.cols; ++j) {
            for (auto k = 0; k < m1.cols; ++k) {
                rm.at(i, j) = rm.at(i, j) + m1.at(i, k) * m2.at(k, j);
            }
        }
    }
    return rm;
}

template <class _Tp>
_Matrix<_Tp> operator*(const _Matrix<_Tp> &m, const Scalar& v)
{
    _Matrix<_Tp> _r(m.shape());
    traverse(m, _r, m.channels(), [&](auto&& _in, auto&& _out) {
        for(auto i = 0; i < m.channels(); ++i) {
            _out[i] = saturate_cast<_Tp>(_in[i] * v[i]);
        }
    });

    return _r;
}

template <class _Tp>
_Matrix<_Tp> operator*(const Scalar& v, const _Matrix<_Tp> &m)
{
    return m * v;
}

template <typename _Tp>
_Matrix<_Tp> operator*(const _Matrix<_Tp>& m, double scale)
{
    assert(m.channels() <= 4);

    _Matrix<_Tp> _r(m.shape());
    traverse(m, _r, 1, [=](auto&& _in, auto&& _out) {
        *_out = saturate_cast<_Tp>((*_in) * scale);
    });

    return _r;
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
_Matrix<_Tp> operator*=(_Matrix<_Tp> &m, const Scalar& v)
{
    assert(m.channels() <= 4);

    traverse(m, m.channels(), [&](auto&& _val) {
        for(auto i = 0; i < m.channels(); ++i) {
            _val[i] = saturate_cast<_Tp>(_val[i] * v[i]);
        }
    });

    return m;
}

template <class _Tp>
_Matrix<_Tp> operator*=(_Matrix<_Tp>& m, double scale)
{
    traverse(m, 1, [=](auto&& _in_out) {
        *_in_out = saturate_cast<_Tp>((*_in_out) * scale);
    });

    return m;
}

template <class _Tp>
_Matrix<_Tp> operator/(const _Matrix<_Tp> &m1, const _Matrix<_Tp> &m2)
{
    _Matrix<_Tp> _r(m1.shape());
    traverse(m1, m2, _r, 1, [](auto&& _in_1, auto&& _in_2, auto&& _out) {
        *_out = saturate_cast<_Tp>((*_in_1) / (*_in_2));
    });

    return _r;
}

template <class _Tp>
_Matrix<_Tp> operator/(const _Matrix<_Tp> &m, const Scalar& v)
{
    assert(m.channels() <= 4);

    _Matrix<_Tp> _r(m.shape());
    traverse(m, _r, m.channels(), [&](auto&& _in, auto&& _out) {
        for(auto i = 0; i < m.channels(); ++i) {
            _out[i] = saturate_cast<_Tp>(_in[i] / v[i]);
        }
    });

    return _r;
}

template <class _Tp>
_Matrix<_Tp> operator/(const Scalar& v, const _Matrix<_Tp> &m)
{
    assert(m.channels() <= 4);

    _Matrix<_Tp> _r(m.shape());
    traverse(m, _r, m.channels(), [&](auto&& _in, auto&& _out) {
        for(auto i = 0; i < m.channels(); ++i) {
            _out[i] = saturate_cast<_Tp>( v[i] / _in[i]);
        }
    });

    return _r;
}

template <class _Tp>
_Matrix<_Tp> operator/(const _Matrix<_Tp>& m, double v)
{
    return m * (1 / v);
}

template <class _Tp>
_Matrix<_Tp> operator/(double v, const _Matrix<_Tp> &m)
{
    _Matrix<_Tp> _r(m.shape());
    traverse(m, _r, 1, [=](auto&& _in, auto&& _out) {
        *_out = saturate_cast<_Tp>(v / (*_in));
    });

    return _r;
}

template <class _Tp>
_Matrix<_Tp> operator/=(_Matrix<_Tp> &m1, const _Matrix<_Tp> &m2)
{
    traverse(m2, m1, 1, [](auto&& _val_2, auto&& _val_1) {
        *_val_1 = saturate_cast<_Tp>((*_val_1) / (*_val_2));
    });

    return m1;
}

template <class _Tp> _Matrix<_Tp> operator/=(_Matrix<_Tp> &m, const Scalar& v)
{
    assert(m.channels() <= 4);

    traverse(m, m.channels(), [&](auto&& _val){
        for(auto i = 0; i < m.channels(); ++i) {
            _val[i] = saturate_cast<_Tp>(_val[i] / v[i]);
        }
    });

    return m;
}

template <class _Tp>
_Matrix<_Tp> operator/=(_Matrix<_Tp>& m, double v)
{
    m *= 1 / v;
    return m;
}

template <class _Tp> _Matrix<_Tp> operator>(const _Matrix<_Tp> &m1, const _Matrix<_Tp> &m2)
{
    assert(m1.shape() == m2.shape());

    alchemy::_Matrix<_Tp> _r(m1.shape());

    traverse(m1, m2, _r, m1.channels(), [&](auto&& _in_1, auto&& _in_2, auto&& _out){
        for(auto i = 0; i < m1.channels(); ++i) {
            _out[i] = _in_1[i] > _in_2 ? std::numeric_limits<_Tp>::max() : std::numeric_limits<_Tp>::min();
        }
    });

    return _r;
}


template <class _Tp> _Matrix<_Tp> operator>(const _Matrix<_Tp> &m, const Scalar& threshold)
{
    assert(m.channels() <= 4);

    alchemy::_Matrix<_Tp> _r(m.shape());

    traverse(m, _r, m.channels(), [&](auto&& _in, auto&& _out){
        for(auto i = 0; i < m.channels(); ++i) {
            _out[i] = _in[i] > threshold[i] ? std::numeric_limits<_Tp>::max() : std::numeric_limits<_Tp>::min();
        }
    });

    return _r;
}
template <class _Tp> _Matrix<_Tp> operator>(const Scalar& threshold, const _Matrix<_Tp> &m)
{
    return m < threshold;
}

template <typename _Tp>
_Matrix<_Tp> operator>(const _Matrix<_Tp>& m, double threshold)
{
    alchemy::_Matrix<_Tp> _r(m.shape());

    traverse(m, _r, 1, [=](auto&& _in, auto&& _out){
        *_out = *_in > threshold ? std::numeric_limits<_Tp>::max() : std::numeric_limits<_Tp>::min();
    });

    return _r;
}

template <typename _Tp>
_Matrix<_Tp> operator>(double threshold, const _Matrix<_Tp>& m)
{
    return m < threshold;
}

template <class _Tp> _Matrix<_Tp> operator>=(const _Matrix<_Tp> &m1, const _Matrix<_Tp>& m2)
{
    return m2 < m1;
}

template <class _Tp> _Matrix<_Tp> operator>=(const _Matrix<_Tp> &m, const Scalar& threshold)
{
    return threshold < m;
}

template <class _Tp> _Matrix<_Tp> operator>=(const Scalar& threshold, const _Matrix<_Tp> &m)
{
    return m < threshold;
}

template <typename _Tp>
_Matrix<_Tp> operator>=(const _Matrix<_Tp>& m, double threshold)
{
    alchemy::_Matrix<_Tp> _r(m.shape());

    traverse(m, _r, 1, [=](auto&& _in, auto&& _out){
        *_out =  *_in >= threshold ? std::numeric_limits<_Tp>::max() : std::numeric_limits<_Tp>::min();
    });

    return _r;
}

template <class _Tp>
_Matrix<_Tp> operator>=(double threshold, const _Matrix<_Tp>& m)
{
    return m <= threshold;
}

template <class _Tp> _Matrix<_Tp> operator<(const _Matrix<_Tp> &m1, const _Matrix<_Tp> &m2)
{
    assert(m1.shape() == m2.shape());

    alchemy::_Matrix<_Tp> _r(m1.shape());

    traverse(m1, m2, _r, m1.channels(), [&](auto&& _in_1, auto&& _in_2, auto&& _out){
        for(auto i = 0; i < m1.channels(); ++i) {
            _out[i] = _in_1[i] < _in_2 ? std::numeric_limits<_Tp>::max() : std::numeric_limits<_Tp>::min();
        }
    });

    return _r;
}
template <class _Tp> _Matrix<_Tp> operator<(const _Matrix<_Tp> &m, const Scalar& threshold)
{
    assert(m.channels() <= 4);

    alchemy::_Matrix<_Tp> _r(m.shape());

    traverse(m, _r, m.channels(), [&](auto&& _in, auto&& _out){
        for(auto i = 0; i < m.channels(); ++i) {
            _out[i] = _in[i] < threshold[i] ? std::numeric_limits<_Tp>::max() : std::numeric_limits<_Tp>::min();
        }
    });

    return _r;
}
template <class _Tp> _Matrix<_Tp> operator<(const Scalar& threshold, const _Matrix<_Tp> &m)
{
    return m > threshold;
}

template <typename _Tp>
_Matrix<_Tp> operator<(const _Matrix<_Tp>& m, double threshold)
{
    alchemy::_Matrix<_Tp> _r(m.shape());

    traverse(m, _r, 1, [=](auto&& _in, auto&& _out){
        *_out =  *_in < threshold ? std::numeric_limits<_Tp>::max() : std::numeric_limits<_Tp>::min();
    });

    return _r;
}

template <typename _Tp>
_Matrix<_Tp> operator < (double threshold, const _Matrix<_Tp>& m)
{
    return m > threshold;
}

template <class _Tp> _Matrix<_Tp> operator<=(const _Matrix<_Tp> &m1, const _Matrix<_Tp>& m2)
{
    return m2 > m1;
}

template <class _Tp> _Matrix<_Tp> operator<=(const _Matrix<_Tp> &m, const Scalar& threshold)
{
    return threshold > m;
}

template <typename _Tp>
_Matrix<_Tp> operator<=(const _Matrix<_Tp>& m, double threshold)
{
    return threshold > m;
}

template <class _Tp> _Matrix<_Tp> operator<=(const Scalar& threshold, const _Matrix<_Tp> &m)
{
    return m > threshold;
}

template <class _Tp>
_Matrix<_Tp> operator<=(double threshold, const _Matrix<_Tp>& m)
{
    return m > threshold;
}

template <typename _Tp>
_Matrix<_Tp> operator==(const _Matrix<_Tp>& m1, const _Matrix<_Tp>& m2)
{
    assert(m1.shape() == m2.shape());

    alchemy::_Matrix<_Tp> _r(m1.shape());

    traverse(m1, m2, _r, 1, [](auto&& _in_1, auto&& _in_2, auto&& _out){
        *_out =  (*_in_1) == (*_in_2) ? std::numeric_limits<_Tp>::max() : std::numeric_limits<_Tp>::min();
    });

    return _r;
}

template <class _Tp> _Matrix<_Tp> operator==(const _Matrix<_Tp> &m, const Scalar& s)
{
    assert(m.channels() <= 4);

    _Matrix<_Tp> _r(m.shape());

    traverse(m, _r, m.channels(), [&](auto&& _in, auto&& _out){
        for(auto i = 0; i < m.channels(); ++i) {
            _out[i] = _in[i] == s[i] ? std::numeric_limits<_Tp>::max() : std::numeric_limits<_Tp>::min();
        }
    });

    return _r;
}
template <class _Tp> _Matrix<_Tp> operator==(const Scalar& s, const _Matrix<_Tp>& m)
{
    return m == s;
}

template <typename _Tp>
_Matrix<_Tp> operator==(const _Matrix<_Tp>& m, double val)
{
    alchemy::_Matrix<_Tp> _r(m.shape());

    traverse(m, _r, 1, [=](auto&& _in, auto&& _out){
        *_out =  *_in == val ? std::numeric_limits<_Tp>::max() : std::numeric_limits<_Tp>::min();
    });

    return _r;
}

template <typename _Tp>
_Matrix<_Tp> operator==(double val, const _Matrix<_Tp>& m)
{
    return m == val;
}

template <typename _Tp>
_Matrix<_Tp> operator!=(const _Matrix<_Tp>& m1, const _Matrix<_Tp>& m2)
{
    assert(m1.shape() == m2.shape());

    alchemy::_Matrix<_Tp> _r(m1.shape());

    traverse(m1, m2, _r, 1, [](auto&& _in_1, auto&& _in_2, auto&& _out){
        *_out =  (*_in_1) != (*_in_2) ? std::numeric_limits<_Tp>::max() : std::numeric_limits<_Tp>::min();
    });

    return _r;
}

template <class _Tp> _Matrix<_Tp> operator!=(const _Matrix<_Tp> &m, const Scalar& s)
{
    assert(m.channels() <= 4);

    _Matrix<_Tp> _r(m.shape());
    traverse(m, _r, m.channels(), [&](auto&& _in, auto&& _out){
        for(auto i = 0; i < m.channels(); ++i){
            _out[i] = _in[i] != s[i] ? std::numeric_limits<_Tp>::max() : std::numeric_limits<_Tp>::min();
        }
    });

    return _r;
}

template <class _Tp> _Matrix<_Tp> operator!=(const Scalar& s, const _Matrix<_Tp> &m)
{
    return m != s;
}

template <typename _Tp>
_Matrix<_Tp> operator!=(const _Matrix<_Tp>& m, double val)
{
    alchemy::_Matrix<_Tp> _r(m.shape());

    traverse(m, _r, 1, [=](auto&& _in, auto&& _out){
        *_out =  *_in != val ? std::numeric_limits<_Tp>::max() : std::numeric_limits<_Tp>::min();
    });

    return _r;
}

template <typename _Tp>
_Matrix<_Tp> operator!=(double val, const _Matrix<_Tp>& m)
{
    return m != val;
}

template<typename _Tp> struct TypeWidth { enum { value = 0 }; };

template<> struct TypeWidth<bool>       { enum { value =  1};};
template<> struct TypeWidth<int8_t>     { enum { value =  4};};
template<> struct TypeWidth<uint8_t>    { enum { value =  3};};
template<> struct TypeWidth<int16_t>    { enum { value =  6};};
template<> struct TypeWidth<uint16_t>   { enum { value =  5};};
template<> struct TypeWidth<int32_t>    { enum { value = 11};};
template<> struct TypeWidth<uint32_t>   { enum { value = 11};};
template<> struct TypeWidth<int64_t>    { enum { value = 20};};
template<> struct TypeWidth<uint64_t>   { enum { value = 20};};

template <typename _Tp> struct CastType{ typedef _Tp cast_type; };

template <> struct CastType<uint8_t>    { typedef int cast_type; };
template <> struct CastType<char>       { typedef int cast_type; };
template <> struct CastType<int8_t>     { typedef int cast_type; };

template <typename _Tp>
std::ostream &operator<<(std::ostream & os, const _Matrix<_Tp> &m)
{
    os << "[";
    for(auto i = 0; i < m.rows; ++i) {
        for(auto j = 0; j < m.cols; ++j) {
            os << "[";
            for(auto k = 0; k < m.channels(); ++k) {
                os << std::setw(TypeWidth<_Tp>::value) << (typename CastType<_Tp>::cast_type)(m.at(i, j, k));

                os << ((k != m.channels() - 1) ? ", " : "]");
            }
        }
        os << ((i + 1 == m.rows) ? ";]\n" : ";\n ");
    }
    return os;
}

///////////////////////////////////////// operations ////////////////////////////////////////////
template<typename _Tp, class Functor>
void traverse(_Matrix<_Tp>& m, std::ptrdiff_t diff, const Functor& callback)
{
    auto _len = m.cols * m.channels();
    for(auto i = 0; i < m.rows; ++i) {
        auto _begin = m.ptr(i);
        auto _end = _begin + _len;
        for(; _begin < _end; _begin += diff) {
            callback(_begin);
        }
    }
};

template<typename _Tp, class Functor>
void traverse(const _Matrix<_Tp>& m, std::ptrdiff_t diff, const Functor& callback)
{
    auto _len = m.cols * m.channels();
    for(auto i = 0; i < m.rows; ++i) {
        auto _begin = m.ptr(i);
        auto _end = _begin + _len;
        for(; _begin < _end; _begin += diff) {
            callback(_begin);
        }
    }
};

template<typename _Tp, class Functor>
void traverse(_Matrix<_Tp>& m1, _Matrix<_Tp>& m2, std::ptrdiff_t diff, const Functor& callback)
{
    assert(m1.shape() == m2.shape());

    auto _len = m1.cols * m1.channels();
    for(auto i = 0; i < m1.rows; ++i) {

        auto _begin_1 = m1.ptr(i);
        auto _end_1 = _begin_1 + _len;
        auto _begin_2 = m2.ptr(i);

        for(; _begin_1 < _end_1; _begin_1 += diff, _begin_2 += diff) {
            callback(_begin_1, _begin_2);
        }
    }
};

template<typename _Tp, class Functor>
void traverse(const _Matrix<_Tp>& m1, _Matrix<_Tp>& m2, std::ptrdiff_t diff, const Functor& callback)
{
    assert(m1.shape() == m2.shape());

    auto _len = m1.cols * m1.channels();
    for(auto i = 0; i < m1.rows; ++i) {

        auto _begin_1 = m1.ptr(i);
        auto _end_1 = _begin_1 + _len;
        auto _begin_2 = m2.ptr(i);

        for(; _begin_1 < _end_1; _begin_1 += diff, _begin_2 += diff) {
            callback(_begin_1, _begin_2);
        }
    }
};

template<typename _Tp, typename _T2, class Functor>
void traverse(_Matrix<_Tp>& m1, _Matrix<_T2>& m2, std::ptrdiff_t diff, const Functor& callback)
{
    assert(m1.shape() == m2.shape());

    auto _len = m1.cols * m1.channels();
    for (auto i = 0; i < m1.rows; ++i) {

        auto _begin_1 = m1.ptr(i);
        auto _end_1 = _begin_1 + _len;
        auto _begin_2 = m2.ptr(i);

        for (; _begin_1 < _end_1; _begin_1 += diff, _begin_2 += diff) {
            callback(_begin_1, _begin_2);
        }
    }
}
template<typename _Tp, typename _T2, class Functor>
void traverse(const _Matrix<_Tp>& m1, _Matrix<_T2>& m2, std::ptrdiff_t diff, const Functor& callback)
{
    assert(m1.shape() == m2.shape());

    auto _len = m1.cols * m1.channels();
    for (auto i = 0; i < m1.rows; ++i) {

        auto _begin_1 = m1.ptr(i);
        auto _end_1 = _begin_1 + _len;
        auto _begin_2 = m2.ptr(i);

        for (; _begin_1 < _end_1; _begin_1 += diff, _begin_2 += diff) {
            callback(_begin_1, _begin_2);
        }
    }
}

template<typename _Tp, class Functor>
void traverse(_Matrix<_Tp>& m1, _Matrix<_Tp>& m2, _Matrix<_Tp>& m3, std::ptrdiff_t diff, const Functor& callback)
{
    assert(m1.shape() == m2.shape());
    assert(m3.shape() == m2.shape());

    auto _len = m1.cols * m1.channels();
    for(auto i = 0; i < m1.rows; ++i) {

        auto _begin_1 = m1.ptr(i);
        auto _begin_2 = m2.ptr(i);
        auto _begin_3 = m3.ptr(i);
        auto _end_1 = _begin_1 + _len;

        for(; _begin_1 < _end_1; _begin_1 += diff, _begin_2 += diff, _begin_3 += diff) {
            callback(_begin_1, _begin_2, _begin_3);
        }
    }
};

template<typename _Tp, class Functor>
void traverse(const _Matrix<_Tp>& m1, const _Matrix<_Tp>& m2, _Matrix<_Tp>& m3, std::ptrdiff_t diff, const Functor& callback)
{
    assert(m1.shape() == m2.shape());
    assert(m3.shape() == m2.shape());

    auto _len = m1.cols * m1.channels();
    for(auto i = 0; i < m1.rows; ++i) {

        auto _begin_1 = m1.ptr(i);
        auto _begin_2 = m2.ptr(i);
        auto _begin_3 = m3.ptr(i);
        auto _end_1 = _begin_1 + _len;

        for(; _begin_1 < _end_1; _begin_1 += diff, _begin_2 += diff, _begin_3 += diff) {
            callback(_begin_1, _begin_2, _begin_3);
        }
    }
};

template<typename _Tp, typename _T2, typename _T3, class Functor>
void traverse(_Matrix<_Tp>& m1, _Matrix<_T2>& m2, _Matrix<_T3>& m3, std::ptrdiff_t diff, const Functor& callback)
{
    assert(m1.shape() == m2.shape());
    assert(m3.shape() == m2.shape());

    auto _len = m1.cols * m1.channels();
    for (auto i = 0; i < m1.rows; ++i) {

        auto _begin_1 = m1.ptr(i);
        auto _begin_2 = m2.ptr(i);
        auto _begin_3 = m3.ptr(i);
        auto _end_1 = _begin_1 + _len;

        for (; _begin_1 < _end_1; _begin_1 += diff, _begin_2 += diff, _begin_3 += diff) {
            callback(_begin_1, _begin_2, _begin_3);
        }
    }
}

template<typename _Tp, typename _T2, typename _T3, class Functor>
void traverse(const _Matrix<_Tp>& m1, const _Matrix<_T2>& m2, _Matrix<_T3>& m3, std::ptrdiff_t diff, const Functor& callback)
{
    assert(m1.shape() == m2.shape());
    assert(m3.shape() == m2.shape());

    auto _len = m1.cols * m1.channels();
    for (auto i = 0; i < m1.rows; ++i) {

        auto _begin_1 = m1.ptr(i);
        auto _begin_2 = m2.ptr(i);
        auto _begin_3 = m3.ptr(i);
        auto _end_1 = _begin_1 + _len;

        for (; _begin_1 < _end_1; _begin_1 += diff, _begin_2 += diff, _begin_3 += diff) {
            callback(_begin_1, _begin_2, _begin_3);
        }
    }
}

template<typename _Tp, class Functor>
void traverse(_Matrix<_Tp>& m1, _Matrix<_Tp>& m2, _Matrix<_Tp>& m3, _Matrix<_Tp>& m4, std::ptrdiff_t diff, const Functor& callback)
{
    assert(m1.shape() == m2.shape());
    assert(m3.shape() == m2.shape());
    assert(m4.shape() == m2.shape());

    auto _len = m1.cols * m1.channels();
    for(auto i = 0; i < m1.rows; ++i) {

        auto _begin_1 = m1.ptr(i);
        auto _begin_2 = m2.ptr(i);
        auto _begin_3 = m3.ptr(i);
        auto _begin_4 = m4.ptr(i);
        auto _end_1 = _begin_1 + _len;

        for(; _begin_1 < _end_1; _begin_1 += diff, _begin_2 += diff, _begin_3 += diff, _begin_4 += diff) {
            callback(_begin_1, _begin_2, _begin_3, _begin_4);
        }
    }
};
template<typename _Tp, class Functor>
void traverse(const _Matrix<_Tp>& m1, const _Matrix<_Tp>& m2, const _Matrix<_Tp>& m3, _Matrix<_Tp>& m4, std::ptrdiff_t diff, const Functor& callback)
{
    assert(m1.shape() == m2.shape());
    assert(m3.shape() == m2.shape());
    assert(m4.shape() == m2.shape());

    auto _len = m1.cols * m1.channels();
    for(auto i = 0; i < m1.rows; ++i) {

        auto _begin_1 = m1.ptr(i);
        auto _begin_2 = m2.ptr(i);
        auto _begin_3 = m3.ptr(i);
        auto _begin_4 = m4.ptr(i);
        auto _end_1 = _begin_1 + _len;

        for(; _begin_1 < _end_1; _begin_1 += diff, _begin_2 += diff, _begin_3 += diff, _begin_4 += diff) {
            callback(_begin_1, _begin_2, _begin_3, _begin_4);
        }
    }
};


template<typename _Tp>
Scalar trace(const _Matrix<_Tp>& m)
{
    return m.trace();
}

template<typename _Tp>
_Matrix<_Tp> abs(const _Matrix<_Tp>& m)
{
    alchemy::_Matrix<_Tp> _r(m.shape());
    traverse(m, _r, 1, [](auto&& _in, auto&& _out){
        *_out = std::abs(*_in);
    });

    return _r;
}

template<typename _Tp> void absdiff(const _Matrix<_Tp>& m1, const _Matrix<_Tp>& m2, _Matrix<_Tp>& dst)
{
    assert(m1.shape() == m2.shape());

    if(dst.shape() != m1.shape()) dst.create(m1.shape());

    traverse(m1, m2, dst, 1, [](auto&& _in_1, auto&&_in_2, auto&& _out){
        *_out = saturate_cast<_Tp>(std::abs(*_in_1 - *_in_2));
    });
}

template<typename _Tp> void absdiff(const Scalar& value, const _Matrix<_Tp>& m, _Matrix<_Tp>& dst)
{
    assert(m.channels() < 5);

    if(dst.shape() != m.shape()) dst.create(m.shape());

    traverse(m, dst, m.channels(), [&](auto&& _in_1,auto&& _out) {
        for(auto i = 0; i < m.channels(); ++i) {
            _out[i] = saturate_cast<_Tp>(std::abs(_in_1[i] - value[i]));
        }
    });
}
template<typename _Tp> void absdiff(const _Matrix<_Tp>& m, const Scalar& value, _Matrix<_Tp>& dst)
{
    absdiff(value, m, dst);
}

template<typename _Tp> void add(const _Matrix<_Tp>& m1, const _Matrix<_Tp>& m2, _Matrix<_Tp>& dst, const Matrix& mask)
{
    if(dst.shape() != m1.shape()) {
        dst.create(m1.shape());
        dst.fill(0);
    }

    if(mask.empty()) {
        traverse(m1, m2, dst, 1, [](auto&& _in_1, auto&& _in_2, auto&& _out) {
            *_out = saturate_cast<_Tp>(*_in_1 + *_in_2);
        });
    }
    else  {
        traverse(m1, m2, mask, dst, 1, [](auto&& _in_1, auto&& _in_2, auto&& _m, auto&& _out) {
            if(*_m) *_out = saturate_cast<_Tp>(*_in_1 + *_in_2);
        });
    }
}


template <typename _Tp>
void addWeighted(const _Matrix<_Tp>&src1, double alpha, const _Matrix<_Tp>&src2, double beta, double gamma, _Matrix<_Tp>&dst)
{
    assert(src1.shape() == src2.shape());

    if(dst.shape() != src1.shape()){
        dst.create(src1.shape());
    }

    auto _len = src1.channels() * src1.cols;
    for (auto i = 0; i < src1.rows; ++i) {
        auto _ptr_1 = src1.ptr(i);
        auto _ptr_2 = src2.ptr(i);
        auto _ptr_3 = dst.ptr(i);
        for (auto j = 0; j < _len; ++j) {
            _ptr_3[j] = saturate_cast<_Tp>(_ptr_1[j] * alpha + _ptr_2[j] * beta + gamma);
        }
    }
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
const _Tp& _MatrixConstIterator<_Tp>::operator[](difference_type i) const
{
    return *reinterpret_cast<const _Tp *>(*this + i);
}

template <typename _Tp>
_MatrixConstIterator<_Tp>& _MatrixConstIterator<_Tp>::operator+=(difference_type ofs)
{
    if (!m_ || !ofs) return *this;
    difference_type ofsb = ofs * esize_;
    ptr_ += ofsb;

    if (ptr_ < start_ || end_ <= ptr_)
    {
        ptr_ -= ofsb;
        seek(ofs, true);
    }
    return *this;
}

template <typename _Tp>
_MatrixConstIterator<_Tp>& _MatrixConstIterator<_Tp>::operator-=(difference_type ofs)
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
_MatrixConstIterator<_Tp> _MatrixConstIterator<_Tp>::operator--(int)
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
_MatrixConstIterator<_Tp> _MatrixConstIterator<_Tp>::operator++(int)
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
bool _MatrixConstIterator<_Tp>::operator < (const _MatrixConstIterator<_Tp>& it) const
{
    return ptr_ < it.ptr_;
}

template <typename _Tp>
bool _MatrixConstIterator<_Tp>::operator > (const _MatrixConstIterator<_Tp>& it) const
{
    return ptr_ > it.ptr_;
}

template <typename _Tp>
void _MatrixConstIterator<_Tp>::seek(difference_type ofs, bool relative)
{
    if (m_->isContinuous()) {
        ptr_ = (relative ? ptr_ : start_) + ofs * esize_;

        if (ptr_ < start_)
            ptr_ = start_;
        else if (ptr_ > end_)
            ptr_ = end_;

        return;
    }

    difference_type row;
    if (relative) {
        difference_type ofs0 = ptr_ - m_->template ptr<uint8_t>();
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
_Tp& _MatrixIterator<_Tp>::operator[](difference_type i) const
{
    return *(*this + i);
}

template <typename _Tp>
_MatrixIterator<_Tp>& _MatrixIterator<_Tp>::operator+=(difference_type ofs)
{
    _MatrixConstIterator<_Tp>::operator+=(ofs);
    return *this;
}

template <typename _Tp>
_MatrixIterator<_Tp>& _MatrixIterator<_Tp>::operator-=(difference_type ofs)
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

#endif //! ALCHEMY_CORE_MATRIX_HPP
