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

#include <stdexcept>

#ifdef __cplusplus

namespace z{

/**
 * @berif 将矩阵初始化为空矩阵
 */
template <class _type>
void _Matrix<_type>::initEmpty()
{
	rows = cols = _size = 0;
	data = datastart = dataend = nullptr;
	refcount = nullptr;
	step = 0;
	chs = 0;
}

/**
 * @berif 真正的创建矩阵，分配内存
 * @attention 所有矩阵数据的分配都应该通过调用该函数实现（调用该函数一般意味着重新创建函数）
 * @param[in] _rows，行数
 * @param[in] _cols，列数
 */
template <class _type>
void _Matrix<_type>::create(int _rows, int _cols, int _chs)
{
	_log_("Matrix create.");

	chs = _chs;
	rows = _rows;
	cols = _cols;
	step = _cols * _chs;
	_size = _rows * _cols;

	// 
	release();

	// 分配
	datastart = data = new _type[_rows * _cols * _chs];
	dataend = data + _size * _chs;
	refcount = new int(1);
}



/**
 * @berif Constructor without params.
 */
template <class _type> _Matrix<_type>::_Matrix()
{
	_log_("Matrix construct without params.");
	initEmpty();
}

template <class _type> _Matrix<_type>::_Matrix(_Size<int> size)
{
	_log_("Matrix construct with params.");
	initEmpty();
	create(size.width, size.height, 1);
}
template <class _type> _Matrix<_type>::_Matrix(_Size<int> size, int _chs)
{
	_log_("Matrix construct with params.");
	initEmpty();
	create(size.width, size.height, _chs);
}
/**
 * @berif Constructor with params.
 * @param[in] _rows，行数
 * @param[in] _cols，列数
 */
template <class _type> _Matrix<_type>::_Matrix(int _rows, int _cols)
{
	_log_("Matrix construct with params.");
	initEmpty();
	create(_rows, _cols, 1);
}

template <class _type> _Matrix<_type>::_Matrix(int _rows, int _cols, int _chs)
{
	_log_("Matrix construct with params.");
	initEmpty();
	create(_rows, _cols, _chs);
}

/**
 * @berif Copying function
 * @attention 这是一个浅复制
 */
template <class _type> _Matrix<_type>::_Matrix(const _Matrix<_type>& m)
	:rows(m.rows), cols(m.cols), data(m.data), refcount(m.refcount),_size(m._size), 
	step(m.step),datastart(m.datastart), dataend(m.dataend), chs(m.chs)
{
	_log_("Matrix copying function.");
	if (refcount)
		refAdd(refcount, 1);
}

/**
 * @berif 控制引用计数的值
 */
template <class _type>
int _Matrix<_type>::refAdd(int *addr, int delta)
{
	int temp = *addr;
	*addr += delta;
	return temp;
}

/**
 * @berif 释放资源
 * @attention 矩阵的资源由该函数控制并释放
 */
template <class _type>
void _Matrix<_type>::release()
{
	if (refcount && refAdd(refcount, -1) == 1) {
		delete[] data;
		data = datastart = dataend = nullptr;
		delete refcount;
		refcount = nullptr;
		_log_("Matrix release.");
	}
}

/**
 * @berif Destructor
 */
template <class _type>
_Matrix<_type>::~_Matrix()
{
	release();
}


/**
 * @berif 形如mat = {1, 2, 3}的赋值方式
 */
template <class _type>
_Matrix<_type>& _Matrix<_type>::operator = (std::initializer_list<_type> li)
{
	if (rows == 0 || cols == 0) {
		create(1, li.size(), 1);
	}

	auto index = li.begin();
	auto end = li.end();
	for (_type * begin = datastart; begin < dataend; ++begin, ++index) {
		if (index < end) {
			*begin = *index;
		}
		else {
			*begin = (_type)0;
		}
	}
	return *this;
}

template <class _type> _Matrix<_type>& _Matrix<_type>::operator += (const _Matrix<_type>& m)
{
	for (size_t i = 0; datastart + i < dataend; ++i) {
		data[i] += m.data[i];
	}
	return (*this);
}
template <class _type> _Matrix<_type>&  _Matrix<_type>::operator -= (const _Matrix<_type>& m)
{
	for (size_t i = 0; datastart + i < dataend; ++i) {
		data[i] -= m.data[i];
	}
	return (*this);
}

/**
 * @berif 将矩阵初始化为0
 */
template <class _type>
void _Matrix<_type>::zeros()
{
	for (size_t i = 0; datastart +i < dataend; ++i) {
		data[i] = 0;
	}
}

/**
 * @berif 重新分配内存并初始化为0
 */
template <class _type>
void _Matrix<_type>::zeros(int _rows, int _cols)
{
	create(_rows, _cols, 1);

	for (size_t i = 0; datastart + i < dataend; ++i) {
		data[i] = 0;
	}
}

/**
 * @berif 将矩阵初始化为1
 */
template <class _type>
void _Matrix<_type>::ones()
{
	for (size_t i = 0; datastart + i < dataend; ++i) {
		data[i] = 1;
	}
}

/**
 * @berif 重新分配内存并初始化为1
 */
template <class _type>
void _Matrix<_type>::ones(int _rows, int _cols)
{
	create(_rows, _cols, 1);

	for (size_t i = 0; datastart + i < dataend; ++i) {
		data[i] = 1;
	}
}


/**
 * @berif 将矩阵初始化为单位矩阵
 */
template <class _type>
void _Matrix<_type>::eye()
{
	if (chs > 1)
		throw runtime_error("channels > 1!!");

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			if (i == j)
				data[i * cols + j] = 1;
			else
				data[i * cols + j] = 0;
		}
	}
}

/**
 * @berif 重新分配内存并初始化为单位矩阵
 */
template <class _type>
void _Matrix<_type>::eye(int _rows, int _cols)
{
	create(_rows, _cols, 1);

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			if (i == j)
				data[i * cols + j] = 1;
			else
				data[i * cols + j] = 0;
		}
	}
}


template <class _type>
void _Matrix<_type>::init(_type _v)
{
	for (size_t i = 0; datastart + i < dataend; ++i)
		data[i] = _v;
}

/**
 * @berif 深度复制函数
 * @param[out] outputMatrix，复制的目的矩阵，会被重新分配内存并复制数据
 */
template <class _type>
void _Matrix<_type>::copyTo(_Matrix<_type> & outputMatrix) const
{
	outputMatrix.create(rows, cols, chs);
	memcpy(outputMatrix.data, data, _size * chs * sizeof(_type));
}

/**
 * @berif 深度复制函数
 * @ret 返回临时矩阵的拷贝
 */
template <class _type>
_Matrix<_type> _Matrix<_type>::clone() const
{
	_Matrix<_type> m;
	copyTo(m);
	return m;
}


/**
 * @berif 赋值函数
 * @attention 这是一个浅复制
 */
template <class _type>
_Matrix<_type>& _Matrix<_type>::operator=(const _Matrix<_type> &m)
{
	_log_("Matrix assignment function.");
	// 防止出现自己给自己复制时候的问题
	if (this != &m) {
		if (m.refcount)
			refAdd(m.refcount, 1);

		// 释放掉左值的内容
		release();

		// 赋值
		chs = m.chs;
		_size = m.size();
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


template <class _type>
_Matrix<_type>& _Matrix<_type>::operator()(_type * InputArray, size_t _size)
{
	create(1, _size, 1);
	for (size_t i = 0; i < _size; ++i)
		data[i] = InputArray[i];

	return *this;
}

template <class _type>
_Matrix<_type>& _Matrix<_type>::operator()(_type * InputArray, int _rows, int _cols)
{
	create(_rows, _cols, 1);
	for (size_t i = 0; datastart + i < dataend; ++i)
		data[i] = InputArray[i];

	return *this;
}

#if defined(OPENCV)
template <class _type>
_Matrix<_type>::operator cv::Mat() const
{
	cv::Mat temp(rows, cols, CV_8UC(chs));

	memcpy(temp.data, data, _size * chs * sizeof(_type));

	return temp;
}
#endif

template <class _type> inline _type* _Matrix<_type>::ptr(int i0, int i1)
{
	if ( (unsigned)i0 >= (unsigned)rows || (unsigned)i1 >= (unsigned)cols) {
		return nullptr;
	}
	return data + i0 * step + i1 * chs;
}

template <class _type> inline const _type* _Matrix<_type>::ptr(int i0, int i1) const
{
	if ((unsigned)i0 >= (unsigned)rows || (unsigned)i1 >= (unsigned)cols) {
		return nullptr;
	}
	return data + i0 * step + i1 * chs;
}

/**
 * @berif 求矩阵的秩
 * m x n矩阵中min(m, n)矩阵的秩
 */
template <class _type>
_type _Matrix<_type>::rank()
{
	_type temp = (_type)0;
	// do something..
	return temp;
}

/**
 * @berif 求矩阵的迹，即对角线元素之和
 * @attention 1、矩阵必须是方阵
 *            2、由于迹是对角线元素之和，所以对于char、short等可能会发生溢出，所以同一改为double
 * m x n矩阵中min(m, n)矩阵的秩
 */
template <class _type>
double _Matrix<_type>::tr()
{
	if (chs != 1)
		throw runtime_error("chs != 1");
	if (rows != cols)
		throw runtime_error("rows != cols");

	_type temp = (_type)0;
	for (int i = 0; i < rows; ++i) {
		temp += (*this)[i][i];
	}
	return temp;
}


/**
* @berif 逆
*/
template <class _type>
_Matrix<_type> _Matrix<_type>::inv()
{
	_Matrix<_type> m(cols, rows);
	// do something..
	return m;
}


/**
 * @berif 转置
 */
template <class _type>
_Matrix<_type>  _Matrix<_type>::t()
{
	_Matrix<_type> m(cols, rows, chs);

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
template <class _type>
_Matrix<_type> _Matrix<_type>::dot(_Matrix<_type> &m)
{
	if (rows != m.rows || cols != m.cols || chs != chs)
		throw runtime_error("rows != m.rows || cols != m.cols || || chs != chs");

	_Matrix<_type> temp(m.rows, m.cols, m.chs);

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
template <class _type>
_Matrix<_type> _Matrix<_type>::cross(_Matrix<_type> &m)
{
	if (rows != 1 || cols != 3 || m.rows != 1 || m.cols != 3 || chs != 0)
		throw runtime_error("rows != 1 || cols != 3 || m.rows != 1 || m.cols != 3 || chs != 0");

	_Matrix<_type> temp(1, 3);

	temp[0][0] = data[1] * m.data[2] - data[2] * m.data[1];
	temp[0][1] = data[2] * m.data[0] - data[0] * m.data[2];
	temp[0][2] = data[0] * m.data[1] - data[1] * m.data[0];

	return temp;
}


template <class _type> void _Matrix<_type>::conv(Matrix &kernel, _Matrix<_type>&dst, bool norm)
{
	if (kernel.rows != kernel.cols || kernel.rows % 2 == 0)
		throw std::runtime_error("size.width != size.height || size.width % 2 == 0");

	if (!dst.equalSize(*this))
		dst.create(rows, cols, chs);

	int *tempValue = new int[chs];
	int zeros = 0;
	int m = kernel.rows / 2, n = kernel.cols / 2;
	const _type * srcPtr = nullptr;
	_type * dstPtr = nullptr;
	int alpha = 0;
	double delta = 0;
	for (size_t i = 0; i < kernel.size(); ++i) {
		delta += kernel.data[i];
	}

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {

			memset(tempValue, 0, chs * sizeof(int));
			zeros = 0;

			for (int ii = 0; ii < kernel.rows; ++ii) {
				for (int jj = 0; jj < kernel.cols; ++jj) {

					// 获取一个像素的地址
					srcPtr = this->ptr(i - m + ii, j - n + jj);

					if (srcPtr) {
						for (int k = 0; k < chs; ++k) {
							tempValue[k] += srcPtr[k] * kernel[ii][jj];
						}
					}
					else {
						zeros++;
					}
				} // !for(jj)
			} // !for(ii)

			alpha = delta - zeros;
			dstPtr = dst.ptr(i, j);

			for (int k = 0; k < chs; ++k) {
				if(norm)
					dstPtr[k] = (_type)(tempValue[k] / alpha);
				else
					dstPtr[k] = (_type)tempValue[k];
			}

		} // !for(j)
	} // !for(i)

	delete[] tempValue;
}

template <class _type> void conv(_Matrix<_type> &src, _Matrix<_type> &dst, Matrix &core)
{
	src.conv(core, dst);
}
/**
 * @berif 重载输出运算符
 */
template <class _type>
std::ostream &operator<<(std::ostream & os, const _Matrix<_type> &item)
{
	os << '[';
	for (int i = 0; i < item.rows; ++i) {
		for (int j = 0; j < item.step; ++j) {
			
			if(sizeof(_type) == 1)
				os << (int)item[i][j];
			else
				os << item[i][j];
			if (item.cols * item.chs != j + 1)
				os << ',';
		}
		if (item.rows != i + 1)
			os << ';' << endl << ' ';
		else
			os << ']' << endl;
	}
	return os;
}

/**
 * @berif 比较两个矩阵是否相等
 */
template <class _type>
bool operator==(const _Matrix<_type> &m1, const _Matrix<_type> &m2)
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
template <class _type>
bool operator!=(const _Matrix<_type> &m1, const _Matrix<_type> &m2)
{
	return !(m1 == m2);
}

/**
 * @berif 矩阵乘法，重载*
 */
template <class _type>
_Matrix<_type> operator*(_Matrix<_type> &m1, _Matrix<_type> &m2)
{
	if (m1.chs != 1 || m2.chs != 1)
		throw runtime_error("m1.chs != 1 || m2.chs != 1");
	if (m1.cols != m2.rows)
		throw runtime_error("m1.cols != m2.rows");

	_Matrix<_type> m(m1.rows, m2.cols, 1);
	m.zeros();

	for (int i = 0; i < m.rows; ++i) {
		for (int j = 0; j < m.cols; ++j) {
			for (int k = 0; k < m1.cols; ++k) {
				m[i][j] += m1[i][k] * m2[k][j];
			}
		}
	}

	return m;
}

/**
 * @berif 矩阵加法，重载+
 */
template <class _type>
_Matrix<_type> operator+(_Matrix<_type> &m1, _Matrix<_type> &m2)
{
	if (m1.cols != m2.cols || m1.rows != m2.rows)
		throw runtime_error("m1.cols != m2.cols || m1.rows != m2.rows");

	_Matrix<_type> temp(m1.rows, m1.cols, m1.chs);

	for (size_t i = 0; datastart + i < dataend; ++i) {
		temp.data[i] = m1.data[i] + m2.data[i];
	}
	return temp;
}

/**
 * @berif 矩阵减法，重载-
 */
template <class _type>
_Matrix<_type> operator-(_Matrix<_type> &m1, _Matrix<_type> &m2)
{
	if (m1.cols != m2.cols || m1.rows != m2.rows)
		throw runtime_error("m1.cols != m2.cols || m1.rows != m2.rows");

	_Matrix<_type> temp(m1.rows, m1.cols, m1.chs);

	for (size_t i = 0; m1.datastart + i < m1.dataend; ++i) {
		temp.data[i] = m1.data[i] - m2.data[i];
	}
	return temp;
}

/**
 * @berif 矩阵数乘，重载*
 */
template <class _type>
_Matrix<_type> operator*(_Matrix<_type> &m, _type delta)
{
	_Matrix<_type> temp(m.rows, m.cols, m.chs);

	for (size_t i = 0; m.datastart + i < m.dataend; ++i) {
		temp.data[i] = m.data[i] * delta;
	}

	return temp;
}

template <class _type>
_Matrix<_type> operator*(_type delta, _Matrix<_type> &m)
{
	return m*delta;
}

/**
 * @berif 矩阵加法，重载+
 */
template <class _type>
_Matrix<_type> operator+(_Matrix<_type> &m, _type delta)
{
	_Matrix<_type> temp(m.rows, m.cols, m.chs);

	for (size_t i = 0; m.datastart + i < m.dataend; ++i) {
		temp.data[i] = m.data[i] + delta;
	}

	return temp;
}
template <class _type>
_Matrix<_type> operator+(_type delta, _Matrix<_type> &m)
{
	return m + delta;
}

/**
 * @berif 矩阵减法，重载-
 */
template <class _type>
_Matrix<_type> operator-(_Matrix<_type> &m, _type delta)
{
	return m + (-delta);
}
template <class _type>
_Matrix<_type> operator-(_type delta, _Matrix<_type> &m)
{
	return m * (-1) + delta;
}


}

#endif // ! __cplusplus

#endif