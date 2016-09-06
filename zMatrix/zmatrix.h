#ifndef _ZMATRIX_H
#define _ZMATRIX_H

#include <stdint.h>
#include <iostream>
#include "config_default.h"

class Matrix {
public:
	// default constructor
	Matrix();
	Matrix(int rows, int cols);

	// 复制构造函数
	Matrix(const Matrix& m);

	~Matrix();

	Matrix& operator = (const Matrix& m);
	Matrix& operator = (std::initializer_list<double>);
	Matrix& operator += (const Matrix& m);

	// 检查这两个函数是否达到了想要的目的
	double* operator[](size_t n);
	const double* operator[](size_t n) const;

	Matrix& operator()(double * InputArray, size_t size);
	Matrix& operator()(double * InputArray, int rows, int cols);

	// 这个函数是否需要两个，const
	double at(int rows, int cols);

	Matrix inv();                        // 逆
	Matrix t();                          // 转置


	//! returns deep copy of the matrix, i.e. the data is copied
	Matrix clone() const;

	//! copies the matrix content to "m".
	// It calls m.create(this->size(), this->type()).
	void copyTo(Matrix & outputMatrix) const;

	//! Matlab-style matrix initialization
	void zeros();
	void ones();
	void eye();
	void zeros(int rows, int cols);
	void ones(int rows, int cols);
	void eye(int rows, int cols);

	//! allocates new matrix data unless the matrix already has specified size and type.
	// previous data is unreferenced if needed.
	void create(int rows, int cols);

	//! returns true if matrix data is NULL
	bool empty() const;
	size_t size() const;

	void release();
	int refAdd(int *addr, int delta);

	// 求秩
	double rank();

	
	Matrix dot(Matrix &m);           // 点乘
	Matrix cross(Matrix &m);         // 叉积
	Matrix conv(Matrix &m);          // 卷积


	int rows, cols; // 行数和列数
	double *data;

private:
	// Size of the Matrix
	size_t _size;

	//! pointer to the reference counter;
	// when matrix points to user-allocated data, the pointer is NULL
	int* refcount;

	void initEmpty();
};


inline bool Matrix::empty() const { return data == nullptr; }
inline size_t Matrix::size() const { return rows*cols; }
// 无越界检查
inline double* Matrix::operator[](size_t n) { return &data[n * cols]; }
inline const double* Matrix::operator[](size_t n) const { return &data[n * cols]; }

// 重载运算符
std::ostream &operator<<(std::ostream & os, const Matrix &item);

bool operator==(const Matrix &m1, const Matrix &m2);
bool operator!=(const Matrix &m1, const Matrix &m2);

Matrix operator*(Matrix &m1, Matrix &m2);
Matrix operator+(Matrix &m1, Matrix &m2);
Matrix operator-(Matrix &m1, Matrix &m2);

#endif  // !_ZMATRIX_H