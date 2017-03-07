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
#include "debug.h"

// 不使用任何宏定义的max和min
#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

namespace z{
//////////////////////////////////////////////////////////////////////////
template<typename _Tp> static inline _Tp saturate_cast(uint8_t v) { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(int8_t v) { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(uint16_t v) { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(int16_t v) { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(uint32_t v) { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(int32_t v) { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(float v) { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(double v) { return _Tp(v); }

// 特化
// such as: -2 -> 0
template <> inline uint8_t saturate_cast<uint8_t>(int8_t v) { return (uint8_t)std::max((int)v, 0); }
template <> inline uint8_t saturate_cast<uint8_t>(uint16_t v) { return (uint8_t)std::min((unsigned)v, (unsigned)(255)); }
template <> inline uint8_t saturate_cast<uint8_t>(int32_t v) { return (uint8_t)((unsigned)v <= 255 ? v : v > 0 ? 255 : 0 ); }
template <> inline uint8_t saturate_cast<uint8_t>(int16_t v) { return (uint8_t)(saturate_cast<uint8_t>(int(v))); }
template <> inline uint8_t saturate_cast<uint8_t>(uint32_t v) { return (uint8_t)(std::min(v, (unsigned)255)); }
// 这个可能有问题
template <> inline uint8_t saturate_cast<uint8_t>(float v) { return (uint8_t)(saturate_cast<uint8_t>((int)v)); }
template <> inline uint8_t saturate_cast<uint8_t>(double v) { return (uint8_t)(saturate_cast<uint8_t>((int)v)); }


//////////////////////////////////////////////////////////////////////////
template<typename _Tp, int n> Vec_<_Tp, n>::Vec_()
{
    for (int i = 0; i < n; ++i)
        data_[i] = _Tp(0);
}
template<typename _Tp, int n> Vec_<_Tp, n>::Vec_(_Tp v0)
{
    assert(n >= 1);
    data_[0] = v0;
}
template<typename _Tp, int n> Vec_<_Tp, n>::Vec_(_Tp v0, _Tp v1)
{
    assert(n >= 2);
    data_[0] = v0, data_[1] = v1;
    for (int i = 2; i < n; ++i) data_[2] = _Tp(0);
} 
template<typename _Tp, int n> Vec_<_Tp, n>::Vec_(_Tp v0, _Tp v1, _Tp v2)
{
    assert(n >= 3);
    data_[0] = v0, data_[1] = v1, data_[2] = v2;
    for (int i = 0; i < n; ++i) data_[i] = _Tp(0);
}
template<typename _Tp, int n> Vec_<_Tp, n>::Vec_(_Tp v0, _Tp v1, _Tp v2, _Tp v3)
{
    assert(n >= 4);
    data_[0] = v0, data_[1] = v1, data_[2] = v2, data_[3] = v3;
    for (int i = 0; i < n; ++i) data_[i] = _Tp(0);
}

template<typename _Tp, int n> Vec_<_Tp, n>::Vec_(const _Tp* vals)
{
    for (int i = 0; i < n; ++i)
        data_[i] = vals[i];
}

template<typename _Tp, int n> Vec_<_Tp, n>::Vec_(const Vec_<_Tp, n>&v) :Vec_<_Tp, n>(v.data_) {  }

template<typename _Tp, int n> Vec_<_Tp, n>& Vec_<_Tp, n >::operator = (std::initializer_list<_Tp> list)
{
    assert(list.size() >= n);

    int idx = 0;
    for (const auto& i : list)
        data_[idx++] = i;

    for (;idx < n; ++idx)
        data_[idx] = _Tp(0);

    return *this;
}

template<typename _Tp, int n> inline const _Tp& Vec_<_Tp, n >::operator[](int i) const
{
    assert(((unsigned)i < (unsigned)n));
    return this->data_[i];
}

template<typename _Tp, int n> inline _Tp& Vec_<_Tp, n >::operator[](int i)
{
    assert(((unsigned)i < (unsigned)n));
    return this->data_[i];
}

template<typename _Tp, int n> inline const _Tp& Vec_<_Tp, n >::operator()(int i) const
{
    assert(((unsigned)i < (unsigned)n));
    return this->data_[i];
}

template<typename _Tp, int n> inline _Tp& Vec_<_Tp, n >::operator()(int i)
{
    assert(((unsigned)i < (unsigned)n));
    return this->data_[i];
}

template<typename _Tp, int n> inline Vec_<_Tp, n> Vec_<_Tp, n>::all(_Tp val)
{
    Vec_<_Tp, n> v;
    for (int i = 0; i < n; ++i)
        v.data_[i] = val;
    return v;
}

template<typename _Tp, int n> inline Vec_<_Tp, n> Vec_<_Tp, n >::zeros()
{
    return all(0);
}
template<typename _Tp, int n> inline Vec_<_Tp, n> Vec_<_Tp, n >::ones()
{
    return all(1);
}

/////////////////////////////////////////////////////////////////////////////////////////////
template<class _Tp>
inline void _Matrix<_Tp>::swap(int32_t i0, int32_t j0, int32_t i1, int32_t j1) {
	_Tp temp;
	for (uint8_t k = 0; k < chs; ++k) {
		temp = ptr(i0, j0)[k];
		ptr(i0, j0)[k] = ptr(i1, j1)[k];
		ptr(i1, j1)[k] = temp;
	}
}

/**
 * @berif 真正的创建矩阵，分配内存
 * @attention 所有矩阵数据的分配都应该通过调用该函数实现（调用该函数一般意味着重新创建函数）
 * @param[in] _rows，行数
 * @param[in] _cols，列数
 */
template <class _Tp>
void _Matrix<_Tp> ::create(Size size, int _chs)
{
    create(size.height, size.width, _chs);
}

template <class _Tp>
void _Matrix<_Tp>::create(int _rows, int _cols, int _chs)
{
    flags = MatrixType<_Tp>::type;
    chssize = sizeof(_Tp);
	chs = _chs;

	rows = _rows;
	cols = _cols;
	step = _cols * chs;
	size_ = _rows * _cols;

	// 
	release();

	// 分配
	datastart = data = new _Tp[size_ * chs];
    //_log_("_Matrix create.");
	dataend = data + size_ * chs;
	refcount = new int(1);
}

template <class _Tp> _Matrix<_Tp>::_Matrix(Size size, int _chs)
{
	create(size.height, size.width, _chs);
}
/**
 * @berif Constructor with params.
 * @param[in] _rows，行数
 * @param[in] _cols，列数
 */
template <class _Tp> _Matrix<_Tp>::_Matrix(int _rows, int _cols, int _chs)
{
	create(_rows, _cols, _chs);
}

/**
 * @berif Copying function
 * @attention 这是一个浅复制
 */
template <class _Tp> _Matrix<_Tp>::_Matrix(const _Matrix<_Tp>& m)
{
    if (this != &m)
        *this = m;
}


/**
 * @berif 赋值函数
 * @attention 这是一个浅复制
 */
template <class _Tp>
_Matrix<_Tp>& _Matrix<_Tp>::operator=(const _Matrix<_Tp> &m)
{
	// 防止出现自己给自己复制时候的问题
	if (this != &m) {
		if (m.refcount)
			refAdd(m.refcount, 1);

		// 释放掉左值的内容
		release();

		// 赋值
        flags = m.flags;
		chs = m.chs;
        chssize = m.chssize;
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

/**
 * @berif 控制引用计数的值
 */
template <class _Tp>
int _Matrix<_Tp>::refAdd(int *addr, int delta)
{
	int temp = *addr;
	*addr += delta;
	return temp;
}

/**
 * @berif 释放资源
 * @attention 矩阵的资源由该函数控制并释放
 */
template <class _Tp>
void _Matrix<_Tp>::release()
{
	if (refcount && (refAdd(refcount, -1) == 1)) {
        //_log_("_Matrix release.");

		delete[] data;
		data = datastart = dataend = nullptr;
		delete refcount;
		refcount = nullptr;
	}
}

/**
 * @berif Destructor
 */
template <class _Tp>
_Matrix<_Tp>::~_Matrix()
{
	release();
}


/**
 * @berif 形如mat = {1, 2, 3}的赋值方式
 */
template <class _Tp>
_Matrix<_Tp>& _Matrix<_Tp>::operator = (std::initializer_list<_Tp> li)
{
	if (rows == 0 || cols == 0)
		create(1, li.size(), 1);

    size_t index = 0;
    for (const auto&i : li)
        data[index++] = i;

    for (; index < size_; ++index)
        data[index] = _Tp(0);

	return *this;
}

template <class _Tp> _Matrix<_Tp>& _Matrix<_Tp>::operator += (const _Matrix<_Tp>& m)
{
	for (size_t i = 0; datastart + i < dataend; ++i)
		data[i] += m.data[i];

	return (*this);
}
template <class _Tp> _Matrix<_Tp>&  _Matrix<_Tp>::operator -= (const _Matrix<_Tp>& m)
{
	for (size_t i = 0; datastart + i < dataend; ++i)
		data[i] -= m.data[i];

	return (*this);
}

/**
 * @berif 将矩阵初始化为0
 */
template <class _Tp>
void _Matrix<_Tp>::zeros()
{
	for (size_t i = 0; datastart + i < dataend; ++i) 
        data[i] = _Tp(0);
}

/**
 * @berif 重新分配内存并初始化为0
 */
template <class _Tp>
void _Matrix<_Tp>::zeros(int _rows, int _cols)
{
	create(_rows, _cols, 1);

	for (size_t i = 0; datastart + i < dataend; ++i)
		data[i] = _Tp(0);
}

/**
 * @berif 将矩阵初始化为1
 */
template <class _Tp>
void _Matrix<_Tp>::ones()
{
	for (size_t i = 0; datastart + i < dataend; ++i) 
        data[i] = _Tp(1);
}

/**
 * @berif 重新分配内存并初始化为1
 */
template <class _Tp>
void _Matrix<_Tp>::ones(int _rows, int _cols)
{
	create(_rows, _cols, 1);

	for (size_t i = 0; datastart + i < dataend; ++i)
		data[i] = _Tp(1);
}


/**
 * @berif 将矩阵初始化为单位矩阵
 */
template <class _Tp>
void _Matrix<_Tp>::eye()
{
    assert((chs == 1) && (rows == cols));

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
            if (i == j)
                data[i * cols + j] = _Tp(1);
            else
                data[i * cols + j] = _Tp(0);
		}
	}
}

/**
 * @berif 重新分配内存并初始化为单位矩阵
 */
template <class _Tp>
void _Matrix<_Tp>::eye(int _rows, int _cols)
{
    assert(_rows == _cols);

	create(_rows, _cols, 1);

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			if (i == j)
				data[i * cols + j] = _Tp(1);
			else
				data[i * cols + j] = _Tp(0);
		}
	}
}

/**
 * @berif 将矩阵所有的值初始化为_v
 */
template <class _Tp>
void _Matrix<_Tp>::init(_Tp _v)
{
	for (size_t i = 0; datastart + i < dataend; ++i)
		data[i] = _Tp(_v);
}

/**
 * @berif 深度复制函数
 * @param[out] outputMatrix，复制的目的矩阵，会被重新分配内存并复制数据
 */
template <class _Tp>
void _Matrix<_Tp>::copyTo(_Matrix<_Tp> & outputMatrix) const
{
	outputMatrix.create(rows, cols, chs);
	memcpy(outputMatrix.data, data, size_ * chs * sizeof(_Tp));
}

/**
 * @berif 深度复制函数
 * @ret 返回临时矩阵的拷贝
 */
template <class _Tp>
_Matrix<_Tp> _Matrix<_Tp>::clone() const
{
	_Matrix<_Tp> m;
	copyTo(m);
	return m;
}

template <class _Tp> 
template <class _Tp2> _Matrix<_Tp>::operator _Matrix<_Tp2>() const
{
	_Matrix<_Tp2> temp(rows, cols, chs);

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols;++j) {
			for (int k = 0; k < chs; ++k) {
                temp.ptr(i, j)[k] = saturate_cast<_Tp2>(ptr(i, j)[k]);
			}
		}
	}
	return temp;
}

template <class _Tp>
_Matrix<_Tp>& _Matrix<_Tp>::operator()(_Tp * InputArray, z::Size size)
{
	create(size.height, size.width, 1);
	for (size_t i = 0; i < _size; ++i) 
        data[i] = InputArray[i];
	return *this;
}

template <class _Tp>
_Matrix<_Tp>& _Matrix<_Tp>::operator()(_Tp * InputArray, int _rows, int _cols)
{
	create(_rows, _cols, 1);
	for (size_t i = 0; datastart + i < dataend; ++i)
		data[i] = InputArray[i];

	return *this;
}

#if defined(OPENCV)
/**
 * @berif 向openCV中的Mat类转换
 */
template <class _Tp>
_Matrix<_Tp>::operator cv::Mat() const
{
	cv::Mat temp(rows, cols, CV_MAKETYPE(flags, chs));

	memcpy(temp.data, data, size_ * chs * sizeof(_Tp));

	return temp;
}
#endif

template <class _Tp> inline _Tp* _Matrix<_Tp>::ptr(int i0)
{
    assert((unsigned)i0 < (unsigned)rows);
	return data + i0 * step;
}


template <class _Tp> inline const _Tp* _Matrix<_Tp>::ptr(int i0) const
{
    assert((unsigned)i0 < (unsigned)rows);
	return data + i0 * step;
}

template <class _Tp> inline _Tp* _Matrix<_Tp>::ptr(int i0, int i1)
{
    assert(!((unsigned)i0 >= (unsigned)rows || (unsigned)i1 >= (unsigned)cols));
	return data + i0 * step + i1 * chs;
}

template <class _Tp> inline const _Tp* _Matrix<_Tp>::ptr(int i0, int i1) const
{
    assert(!((unsigned)i0 >= (unsigned)rows || (unsigned)i1 >= (unsigned)cols));
	return data + i0 * step + i1 * chs;
}


template <class _Tp> _Tp* _Matrix<_Tp>::ptr(int i0, int i1, int i2)
{
    assert(!((unsigned)i0 >= (unsigned)rows || (unsigned)i1 >= (unsigned)cols || (unsigned)i2 >= (unsigned)chs));
    return data + i0 * step + i1 * chs + i2;
}

template <class _Tp> const _Tp* _Matrix<_Tp>::ptr(int i0, int i1, int i2) const
{
    assert(!((unsigned)i0 >= (unsigned)rows || (unsigned)i1 >= (unsigned)cols || (unsigned)i2 >= (unsigned)chs));
    return data + i0 * step + i1 * chs + i2;
}

template <class _Tp>
template<typename _T2> _T2* _Matrix<_Tp>::ptr(int i0)
{
    assert(!((unsigned)i0 >= (unsigned)rows));
    return (_T2 *)(data + i0 * step);
}

template <class _Tp>
template<typename _T2> const _T2* _Matrix<_Tp>::ptr(int i0) const
{
    assert(!((unsigned)i0 >= (unsigned)rows));
    return (const _T2 *)(data + i0 * step);
}

template<typename _Tp>
template<typename _T2> _T2* _Matrix<_Tp>::ptr(int i0, int i1)
{
    assert(!((unsigned)i0 >= (unsigned)rows || (unsigned)i1 >= (unsigned)cols));
    return (_T2 *)(data + i0 * step + i1 * chs);
}

template <class _Tp>
template<typename _T2> const _T2* _Matrix<_Tp>::ptr(int i0, int i1) const
{
    assert(!((unsigned)i0 >= (unsigned)rows || (unsigned)i1 >= (unsigned)cols));
    return (const _T2 *)(data + i0 * step + i1 * chs);
}

template <class _Tp>
template<typename _T2> _T2* _Matrix<_Tp>::ptr(int i0, int i1, int i2)
{
    assert(!((unsigned)i0 >= (unsigned)rows || (unsigned)i1 >= (unsigned)cols || (unsigned)i2 >= (unsigned)chs));
    return (_T2 *)(data + i0 * step + i1 * chs + i2);
}

template <class _Tp>
template<typename _T2> const _T2* _Matrix<_Tp>::ptr(int i0, int i1, int i2) const 
{
    assert(!((unsigned)i0 >= (unsigned)rows || (unsigned)i1 >= (unsigned)cols || (unsigned)i2 >= (unsigned)chs));
    return (const _T2 *)(data + i0 * step + i1 * chs + i2);
}

/**
 * @berif 求矩阵的秩
 * m x n矩阵中min(m, n)矩阵的秩
 */
template <class _Tp>
_Tp _Matrix<_Tp>::rank()
{
    _Tp temp = (_Tp)0;
	// do something..
	return temp;
}

/**
 * @berif 求矩阵的迹，即对角线元素之和
 * @attention 1、矩阵必须是方阵
 *            2、由于迹是对角线元素之和，所以对于char、short等可能会发生溢出，所以同一改为double
 * m x n矩阵中min(m, n)矩阵的秩
 */
template <class _Tp>
double _Matrix<_Tp>::tr()
{
    assert(chs == 1 && rows == cols);

    _Tp temp = _Tp(0);
	for (int i = 0; i < rows; ++i) {
		temp += (*this)[i][i];
	}
	return temp;
}


/**
 * @berif 逆
 */
template <class _Tp>
_Matrix<_Tp> _Matrix<_Tp>::inv()
{
	_Matrix<_Tp> m(cols, rows);
	// do something..
	return m;
}


/**
 * @berif 转置
 */
template <class _Tp>
_Matrix<_Tp>  _Matrix<_Tp>::t()
{
	_Matrix<_Tp> m(cols, rows, chs);

	for (int i = 0; i < m.rows; ++i) {
		for (int j = 0; j < m.cols; ++j) {
			for (int k = 0; k < chs; ++k) {
				m[i][j * chs + k] = (*this)[j][i * chs + k];
			}
		}
	}
	return m;
}

/**
 * @berif 点乘
 */
template <class _Tp>
_Matrix<_Tp> _Matrix<_Tp>::dot(_Matrix<_Tp> &m)
{
	if (rows != m.rows || cols != m.cols || chs != chs)
		_log_("rows != m.rows || cols != m.cols || || chs != chs");

	_Matrix<_Tp> temp(m.rows, m.cols, m.chs);

	for (size_t i = 0; datastart + i < dataend; ++i) {
		temp.data[i] = data[i] * m.data[i];
	}

	return temp;
}

/**
 * @berif 叉乘
 * @attention C = cross(A,B) returns the cross product of the vectors
 *            A and B.  That is, C = A x B.  A and B must be 3 element
 *            vectors.
 */
template <class _Tp>
_Matrix<_Tp> _Matrix<_Tp>::cross(_Matrix<_Tp> &m)
{
	if (rows != 1 || cols != 3 || m.rows != 1 || m.cols != 3 || chs != 0)
		_log_("rows != 1 || cols != 3 || m.rows != 1 || m.cols != 3 || chs != 0");

	_Matrix<_Tp> temp(1, 3);

	temp[0][0] = data[1] * m.data[2] - data[2] * m.data[1];
	temp[0][1] = data[2] * m.data[0] - data[0] * m.data[2];
	temp[0][2] = data[0] * m.data[1] - data[1] * m.data[0];

	return temp;
}

/**
 * @berif 卷积运算
 *
 * @param[in] kernel， 卷积核
 * @param[out] dst，结果
 * @param[in] norm， 是否对卷积结果归一化
 */
template <typename _Tp>
template <typename _T2> _Matrix<_Tp> _Matrix<_Tp>::conv(const _Matrix<_T2> &kernel, bool norm) const
{
    assert(kernel.cols == kernel.rows && kernel.rows % 2 != 0);

    z::_Matrix<_Tp> dst(rows, cols, chs);

    double *tempValue = new double[chs];                        // 保存卷积未归一化前的中间变量
    int m = kernel.rows / 2, n = kernel.cols / 2;
    _T2 alpha = 0, delta = 0, zeros = 0;                     // 边缘处理和归一化

    if (norm) {
        for (size_t i = 0; i < kernel.size_; ++i) {
            delta += kernel.data[i];
        }
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {

            memset(tempValue, 0, chs * sizeof(double));
            zeros = 0;

            for (int ii = 0; ii < kernel.rows; ++ii) {
                for (int jj = 0; jj < kernel.cols; ++jj) {
                    auto _i = i - m + ii;
                    auto _j = j - n + jj;
                    if ((unsigned)_i < (unsigned)rows && (unsigned)_j < (unsigned)cols) {
                        for (int k = 0; k < chs; ++k) {
                            tempValue[k] += ptr(_i, _j)[k] * kernel[ii][jj];
                        }
                    }
                    else {
                        zeros += kernel.ptr(ii, jj)[0];
                    }
                } // !for(jj)
            } // !for(ii)

            if (norm) {
                alpha = delta - zeros;
                for (int k = 0; k < chs; ++k) dst.ptr(i, j)[k] = saturate_cast<_Tp>(tempValue[k] / alpha);
            }
            else {
                for (int k = 0; k < chs; ++k) dst.ptr(i, j)[k] = saturate_cast<_Tp>(tempValue[k]);
            }
        } // !for(j)
    } // !for(i)

    delete[] tempValue;

    return dst;
}

template <class _Tp> void conv(_Matrix<_Tp> &src, _Matrix<_Tp> &dst, Matrix64f &core)
{
	src.conv(core, dst);
}
/**
 * @berif 重载输出运算符
 */
template <class _Tp>
std::ostream &operator<<(std::ostream & os, const _Matrix<_Tp> &item)
{
	os << '[';
	for (int i = 0; i < item.rows; ++i) {
		for (int j = 0; j < item.step; ++j) {
			
			if(sizeof(_Tp) == 1)
				os << (int)item[i][j];
			else
				os << item[i][j];
			if (item.cols * item.chs != j + 1)
				os << ", ";
		}
		if (item.rows != i + 1)
			os << ';' << std::endl << ' ';
		else
			os << ']' << std::endl;
	}
	return os;
}

/**
 * @berif 比较两个矩阵是否相等
 * @attention 浮点数的比较
 */
template <class _Tp>
bool operator==(const _Matrix<_Tp> &m1, const _Matrix<_Tp> &m2)
{
	// 1、没有分配内存的矩阵比较，没有经过create()
	if (m1.data == nullptr && m1.data == m2.data) {
		return true;
	}
	// 2、分配内存
	else if (m1.data != nullptr) {
		// 内存地址相等，引用，相等
		if (m1.data == m2.data)
			return true;
		// 地址不相等, 行列相等的前提下，元素相等
		else {
			if (m1.cols == m2.cols && m1.rows == m2.rows && m1.chs == m2.chs) {
				int i = 0;
				for (; m1.datastart + i < m1.dataend; ++i) {
					if (m1.data[i] != m2.data[i])
						break;
				}
				if (i == m1.size())
					return true;
			}
		}
	}
	return false;
}

/**
 * @berif 比较两个矩阵是否不相等
 */
template <class _Tp>
bool operator!=(const _Matrix<_Tp> &m1, const _Matrix<_Tp> &m2)
{
	return !(m1 == m2);
}

/**
 * @berif 矩阵乘法，重载*
 */
template <class _Tp>
_Matrix<_Tp> operator*(_Matrix<_Tp> &m1, _Matrix<_Tp> &m2)
{
    assert(m1.cols == m2.rows && m1.chs == 1 && m2.chs == 1);

	_Matrix<_Tp> m(m1.rows, m2.cols, 1);
	m.zeros();

	for (int i = 0; i < m.rows; ++i) {
		for (int j = 0; j < m.cols; ++j) {
			for (int k = 0; k < m1.cols; ++k) {
				m[i][j] = saturate_cast<_Tp>(m[i][j] + m1[i][k] * m2[k][j]);
			}
		}
	}

	return m;
}

/**
 * @berif 矩阵加法，重载+
 */
template <class _Tp>
_Matrix<_Tp> operator+(_Matrix<_Tp> &m1, _Matrix<_Tp> &m2)
{
    assert(m1.cols == m2.cols && m1.rows == m2.rows && m1.chs == m2.chs);

	_Matrix<_Tp> temp(m1.rows, m1.cols, m1.chs);

	for (size_t i = 0; m1.datastart + i < m1.dataend; ++i) {
		temp.data[i] = saturate_cast<_Tp>(m1.data[i] + m2.data[i]);
	}
	return temp;
}

/**
 * @berif 矩阵减法，重载-
 */
template <class _Tp>
_Matrix<_Tp> operator-(_Matrix<_Tp> &m1, _Matrix<_Tp> &m2)
{
    assert(m1.cols == m2.cols && m1.rows == m2.rows && m1.chs == m2.chs);

	_Matrix<_Tp> temp(m1.rows, m1.cols, m1.chs);

	for (size_t i = 0; m1.datastart + i < m1.dataend; ++i) {
		temp.data[i] = saturate_cast<_Tp>(m1.data[i] - m2.data[i]);
	}
	return temp;
}

/**
 * @berif 矩阵数乘，重载*
 */
template <class _Tp>
_Matrix<_Tp> operator*(_Matrix<_Tp> &m, _Tp delta)
{
	_Matrix<_Tp> temp(m.rows, m.cols, m.chs);

	for (size_t i = 0; m.datastart + i < m.dataend; ++i) {
		temp.data[i] = m.data[i] * delta;
	}

	return temp;
}

template <class _Tp>
_Matrix<_Tp> operator*(_Tp delta, _Matrix<_Tp> &m)
{
	return m*delta;
}

/**
 * @berif 矩阵加法，重载+
 */
template <class _Tp>
_Matrix<_Tp> operator+(_Matrix<_Tp> &m, _Tp delta)
{
	_Matrix<_Tp> temp(m.rows, m.cols, m.chs);

	for (size_t i = 0; m.datastart + i < m.dataend; ++i) {
		temp.data[i] = saturate_cast<_Tp>(m.data[i] + delta);
	}

	return temp;
}
template <class _Tp>
_Matrix<_Tp> operator+(_Tp delta, _Matrix<_Tp> &m)
{
	return m + delta;
}

/**
 * @berif 矩阵减法，重载-
 */
template <class _Tp>
_Matrix<_Tp> operator-(_Matrix<_Tp> &m, _Tp delta)
{
	return m + (-delta);
}
template <class _Tp>
_Matrix<_Tp> operator-(_Tp delta, _Matrix<_Tp> &m)
{
	return m * (-1) + delta;
}

template <class _T1, class _T2> _Matrix<_T1> operator>(const _Matrix<_T1> &m, _T2 threshold)
{
    _Matrix<_T1> temp(m.rows, m.cols, m.chs);
    temp.zeros();

    for (size_t i = 0; m.datastart + i < m.dataend; ++i) {
        if (m.data[i] > threshold)
            temp.data[i] = 255;
    }

    return temp;
}


template <class _T1, class _T2> _Matrix<_T1> operator<(const _Matrix<_T1> &m, _T2 threshold)
{
    _Matrix<_T1> temp(m.rows, m.cols, m.chs);
    temp.zeros();

    for (size_t i = 0; m.datastart + i < m.dataend; ++i) {
        if (m.data[i] < threshold)
            temp.data[i] = 255;
    }

    return temp;
}

template <class _T1, class _T2> _Matrix<_T1> operator>=(const _Matrix<_T1> &m, _T2 threshold)
{
    _Matrix<_T1> temp(m.rows, m.cols, m.chs);
    temp.zeros();

    for (size_t i = 0; m.datastart + i < m.dataend; ++i) {
        if (m.data[i] >= threshold)
            temp.data[i] = 255;
    }

    return temp;
}

template <class _T1, class _T2> _Matrix<_T1> operator<=(const _Matrix<_T1> &m, _T2 threshold)
{
    _Matrix<_T1> temp(m.rows, m.cols, m.chs);
    temp.zeros();

    for (size_t i = 0; m.datastart + i < m.dataend; ++i) {
        if (m.data[i] <= threshold)
            temp.data[i] = 255;
    }

    return temp;
}

/////////////////////////////////////////_Complex_////////////////////////////////////////////
template <class _Tp> inline _Complex_<_Tp>::_Complex_() :re(0), im(0) {}
template <class _Tp> inline _Complex_<_Tp>::_Complex_(_Tp _re, _Tp _im) : re(_re), im(_im) {}
template <class _Tp> inline _Complex_<_Tp>::_Complex_(const _Complex_&c) : re(c.re), im(c.im) {}
template <class _Tp> _Complex_<_Tp> & _Complex_<_Tp>::operator=(const _Complex_ &c) { re = c.re; im = c.im; return *this; };

template <class _Tp> bool operator ==(const _Complex_<_Tp> & c1, const _Complex_<_Tp> &c2)
{
	return (c1.re == c2.re && c1.im == c2.im);
}
template <class _Tp> bool operator !=(const _Complex_<_Tp> & c1, const _Complex_<_Tp> &c2)
{
	return !(c1 == c2);
}

template <class _Tp> _Complex_<_Tp> operator*(const _Complex_<_Tp> & c1, const _Complex_<_Tp> & c2)
{
	return _Complex_<_Tp>(c1.re * c2.re - c1.im * c2.im, c1.im * c2.re + c1.re * c2.im);
}

template <class _Tp> _Complex_<_Tp> operator+(const _Complex_<_Tp> & c1, const _Complex_<_Tp> &c2)
{
	return _Complex_<_Tp>(c1.re + c2.re, c1.im + c2.im);
}

template <class _Tp> _Complex_<_Tp> operator-(const _Complex_<_Tp> & c1, const _Complex_<_Tp> &c2)
{
	return _Complex_<_Tp>(c1.re - c2.re, c1.im - c2.im);
}

template <class _Tp> _Complex_<_Tp>& _Complex_<_Tp>::operator+=(const _Complex_<_Tp> & c)
{
	re += c.re;
	im += c.im;
	return *this;
}

template <class _Tp> _Complex_<_Tp>& _Complex_<_Tp>::operator-=(const _Complex_<_Tp> & c)
{
	re -= c.re;
	im -= c.im;
	return *this;
}

template <class _Tp> std::ostream & operator<<(std::ostream & os, const _Complex_<_Tp> & c)
{
	os << "(" << c.re << ", " << c.im << ")";
}

}

#endif // ! _OPERATIONS_HPP
