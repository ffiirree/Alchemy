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
#include "config_default.h"

#if defined(OPENCV)
#include <opencv2\core.hpp>
#endif

#ifdef __cplusplus

template <class _type> class _Matrix;

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
	_Matrix(const _Matrix<_type>& m);
	~_Matrix();

	//! allocates new matrix data unless the matrix already has specified size and type.
	// previous data is unreferenced if needed.
	void create(int rows, int cols);

	void release();
	int refAdd(int *addr, int delta);

	_Matrix<_type>& operator = (const _Matrix<_type>& m);
	_Matrix<_type>& operator = (std::initializer_list<_type>);
	_Matrix<_type>& operator += (const _Matrix<_type>& m);


	// 这个函数是否需要两个，const
	_type at(int rows, int cols);

	// 检查这两个函数是否达到了想要的目的
	inline _type* operator[](size_t n) { return &data[n * cols]; }
	inline const _type* operator[](size_t n) const { return &data[n * cols]; }

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

	//! returns true if matrix data is NULL
	inline bool empty() const { return data == nullptr; }
	inline size_t size() const { return _size; }

	_Matrix<_type> inv();                            // 逆
	_Matrix<_type> t();                              // 转置

	_type rank();                                    // 求秩
	_type tr();                                      // 迹
	
	_Matrix<_type> dot(_Matrix<_type> &m);           // 点乘
	_Matrix<_type> cross(_Matrix<_type> &m);         // 叉积
	_Matrix<_type> conv(Matrix64f &m);              // 卷积


	int rows, cols;
	_type *data;

private:
	size_t _size;

	//! pointer to the reference counter;
	// when matrix points to user-allocated data, the pointer is NULL
	int* refcount;

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



template <class _type> _Matrix<_type> conv(_Matrix<_type> &m, Matrix &core);

#endif // !__cplusplus

#include "operations.hpp"

#endif  // !_ZMATRIX_H