#include "zmatrix.h"

using namespace std;

/**
 * @berif 将矩阵初始化为空矩阵
 */
void Matrix::initEmpty()
{
	rows = cols = _size = 0;
	data = nullptr;
	refcount = nullptr;
}

/**
 * @berif 真正的创建矩阵，分配内存
 * @attention 所有矩阵数据的分配都应该通过调用该函数实现（调用该函数一般意味着重新创建函数）
 * @param[in] _rows，行数
 * @param[in] _cols，列数
 */
void Matrix::create(int _rows, int _cols)
{
	_log_("Matrix create.");

	rows = _rows;
	cols = _cols;
	_size = rows * cols;

	// 
	release();

	// 分配
	data = new double[_rows * _cols];
	refcount = new int(1);
}

/**
 * @berif 控制引用计数的值
 */
int Matrix::refAdd(int *addr, int delta)
{
	int temp = *addr;
	*addr += delta;
	return temp;
}

/**
 * @berif 释放资源
 * @attention 矩阵的资源由该函数控制并释放
 */
void Matrix::release()
{
	if (refcount && refAdd(refcount, -1) == 1) {
		delete[] data;
		data = nullptr;
		delete refcount;
		refcount = nullptr;
		_log_("Matrix release.");
	}
}

/**
 * @berif 无参构造函数
 */
Matrix::Matrix()
{
	_log_("Matrix construct without params.");
	initEmpty();
}

/**
 * @berif 构造函数
 * @param[in] _rows，行数
 * @param[in] _cols，列数
 */
Matrix::Matrix(int _rows, int _cols)
{
	_log_("Matrix construct with params.");
	initEmpty();
	create(_rows, _cols);
}
/**
 * @berif 拷贝函数
 * @attention 这是一个浅复制
 */
Matrix::Matrix(const Matrix& m)
	:rows(m.rows),cols(m.cols),data(m.data),refcount(m.refcount)
{
	_log_("Matrix copying function.");
	if (refcount)
		refAdd(refcount, 1);
}

/**
 * @berif 将矩阵初始化为0
 */
void Matrix::zeros()
{
	for (int i = 0; i < _size; ++i) {
		data[i] = 0;
	}
}

/**
 * @berif 重新分配内存并初始化为0
 */
void Matrix::zeros(int _rows, int _cols)
{
	create(_rows, _cols);

	for (int i = 0; i < _size; ++i) {
		data[i] = 0;
	}
}

/**
 * @berif 将矩阵初始化为1
 */
void Matrix::ones()
{
	for (int i = 0; i < _size; ++i) {
		data[i] = 1;
	}
}

/**
 * @berif 重新分配内存并初始化为1
 */
void Matrix::ones(int _rows, int _cols)
{
	create(_rows, _cols);

	for (int i = 0; i < _size; ++i) {
		data[i] = 1;
	}
}


/**
 * @berif 将矩阵初始化为单位矩阵
 */
void Matrix::eye()
{
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
void Matrix::eye(int _rows, int _cols)
{
	create(_rows, _cols);

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			if(i  == j)
				data[i * cols + j] = 1;
			else
				data[i * cols + j] = 0;
		}
	}
}

/**
 * @berif 深度复制函数
 * @param[out] outputMatrix，复制的目的矩阵，会被重新分配内存并复制数据
 */
void Matrix::copyTo(Matrix & outputMatrix) const
{
	outputMatrix.create(rows, cols);
	memcpy(outputMatrix.data, data, _size*sizeof(double));
}

/**
 * @berif 深度复制函数
 * @ret 返回临时矩阵的拷贝
 */
Matrix Matrix::clone() const
{
	Matrix m;
	copyTo(m);
	return m;
}


/**
 * @berif 析构函数
 */
Matrix::~Matrix()
{
	release();
}

/**
* @berif 赋值函数
* @attention 这是一个浅复制
*/
Matrix& Matrix::operator=(const Matrix &m)
{
	_log_("Matrix assignment function.");
	// 防止出现自己给自己复制时候的问题
	if (this != &m) {
		if (m.refcount)
			refAdd(m.refcount, 1);

		// 释放掉左值的内容
		release();

		// 赋值
		data = m.data;
		refcount = m.refcount;
		rows = m.rows;
		cols = m.cols;
	}

	return *this;
}


/**
 * @berif 形如mat = {1, 2, 3}的赋值方式
 */
Matrix& Matrix::operator = (std::initializer_list<double> li)
{
	if (rows != 0 && cols != 0) {
		create(rows, cols);
	}
	else {
		create(1, li.size());
	}
	
	auto index = li.begin();
	auto end = li.end();
	for (int i = 0; i < _size; ++i, ++index) {
		if (index < end) {
			data[i] = *index;
		}
		else {
			data[i] = 0.0f;
		}
	}
	return *this;
}

Matrix& Matrix::operator()(double * InputArray, size_t _size)
{
	create(1, _size);
	for (int i = 0; i < _size; ++i)
		data[i] = InputArray[i];

	return *this;
}

Matrix& Matrix::operator()(double * InputArray, int _rows, int _cols)
{
	create(_rows, _cols);
	for (int i = 0; i < _size; ++i)
		data[i] = InputArray[i];

	return *this;
}







/**
 * @berif 求矩阵的秩
 * m x n矩阵中min(m, n)矩阵的秩
 */
double	Matrix::rank()
{
	double temp = 0.0f;
	int min_m_n = rows < cols ? rows : cols;
	for (int i = 0; i < min_m_n; ++i) {
		for (int j = 0; j < min_m_n; ++j) {
			if (i == j) {
				temp += data[i * cols + j];
			}
		}
	}
	return temp;
}


/**
 * @berif 重载输出运算符
 */
ostream &operator<<(ostream & os, const Matrix &item)
{
	os << '[';
	for (int i = 0; i < item.rows; ++i) {
		for (int j = 0; j < item.cols; ++j) {
			os << item.data[i*item.cols + j];
			if (item.cols != j + 1)
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
bool operator==(const Matrix &m1, const Matrix &m2)
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
			if (m1.cols == m2.cols && m1.rows == m2.rows) {
				int i = 0;
				for (; i < m1.size(); ++i) {
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

bool operator!=(const Matrix &m1, const Matrix &m2)
{
	return !(m1 == m2);
}


Matrix operator*(Matrix &m1, Matrix &m2)
{
	try {
		if (m1.cols != m2.rows) {
			throw;
		}
	}
	catch (exception){
		_log_("矩阵1的列不等于矩阵2的行数");
		return Matrix();
	}

	Matrix m(m1.rows, m2.cols);
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

Matrix operator+(Matrix &m1, Matrix &m2)
{
	try {
		if (m1.cols != m2.cols || m1.rows != m2.rows) {
			throw;
		}
	}
	catch (exception) {
		_log_("m1.cols != m2.cols || m1.rows != m2.rows");
		return Matrix();
	}

	Matrix temp(m1.rows, m1.cols);

	for (int i = 0; i < temp.rows; ++i) {
		for (int j = 0; j < temp.cols; ++j) {
			temp[i][j] = m1[i][j] + m2[i][j];
		}
	}
	return temp;
}

Matrix operator-(Matrix &m1, Matrix &m2)
{
	try {
		if (m1.cols != m2.cols || m1.rows != m2.rows) {
			throw;
		}
	}
	catch (exception) {
		_log_("m1.cols != m2.cols || m1.rows != m2.rows");
		return Matrix();
	}

	Matrix temp(m1.rows, m1.cols);

	for (int i = 0; i < temp.rows; ++i) {
		for (int j = 0; j < temp.cols; ++j) {
			temp[i][j] = m1[i][j] - m2[i][j];
		}
	}
	return temp;
}

/**
 * @berif 逆
 */
Matrix Matrix::inv()
{
	Matrix m(cols,rows);
	
	return m;
}


/**
 * @berif 转置
 */
Matrix  Matrix::t()
{
	try {
		if (rows != cols) {
			throw;
		}
	}
	catch (exception) {
		_log_("rows != cols");
		return Matrix();
	}

	Matrix m(cols, rows);
	for (int i = 0; i < m.rows; ++i) {
		for (int j = 0; j < m.cols; ++j) {
			m[i][j] = (*this)[j][i];
		}
	}
	return m;
}

/**
 * @berif 点乘
 */
Matrix Matrix::dot(Matrix &m)
{
	Matrix temp(3, 3);
	return temp;
}

/**
 * @berif 叉乘
 * @attention C = cross(A,B) returns the cross product of the vectors
    A and B.  That is, C = A x B.  A and B must be 3 element
    vectors.
 */
Matrix Matrix::cross(Matrix &m)
{
	try {
		if (rows != 1 || cols != 3 || m.rows != 1 || m.cols != 3) {
			throw;
		}
	}
	catch (exception) {
		_log_("矩阵不符合运算规范");
		return Matrix();
	}

	Matrix temp(1, 3);

	temp[0][0] = data[1] * m.data[2] - data[2] * m.data[1];
	temp[0][1] = data[2] * m.data[0] - data[0] * m.data[2];
	temp[0][2] = data[0] * m.data[1] - data[1] * m.data[0];

	return temp;
}

/**
 * @berif 卷积，暂时只限于卷积核为3*3
 */
Matrix Matrix::conv(Matrix &m)
{
	try {
		if (m.rows != m.cols || m.rows % 2 == 0) {
			throw;
		}
	}
	catch (exception) {
		_log_("矩阵不符合运算规范");
		return Matrix();
	}

	Matrix temp(rows, cols);
	int depth = m.rows / 2;

	for (int i = 0; i < temp.rows; ++i) {
		for (int j = 0; j < temp.cols; ++j) {
			temp[i][j] = (*this).at(i - 1, j - 1) * m[0][0] + (*this).at(i - 1, j) * m[0][1] + (*this).at(i - 1, j + 1) * m[0][2]
				+ (*this).at(i, j - 1) * m[1][0] + (*this).at(i, j) * m[1][1] + (*this).at(i, j + 1) * m[1][2]
				+ (*this).at(i + 1, j - 1) * m[2][0] + (*this).at(i + 1, j) * m[2][1] + (*this).at(i + 1, j + 1) * m[2][2];
		}
	}

	return temp;
}

/**
 * @berif 带有越界检查
 */
double Matrix::at(int _rows, int _cols)
{
	if (_rows < 0 || _cols < 0 || _rows >= rows || _cols >= cols) {
		return 0.0;
	}
	else {
		return (*this)[_rows][_cols];
	}
}