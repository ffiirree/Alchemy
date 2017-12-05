#ifndef _ZCORE_MATRIX_H
#define _ZCORE_MATRIX_H

#include <cstdint>
#include <functional>
#include "traits.h"
#include "config.h"
#include "types.h"


#if defined(USE_OPENCV)
#include "opencv2/opencv.hpp"
#endif

namespace z {
/////////////////////////////////////////_Matrix////////////////////////////////////////////
template<typename _Tp> class _MatrixConstIterator;
template<typename _Tp> class _MatrixIterator;

/**
 * @brief
 */
template <typename _Tp> class _Matrix {
public:
    typedef _Tp value_type;
    typedef _MatrixIterator<_Tp> iterator;
    typedef _MatrixConstIterator<_Tp> const_iterator;

    typedef _Tp& reference;
    typedef const _Tp& const_reference;

    /**
     * @brief Constructor
     */
    _Matrix() = default;

    /**
     * @overload
     * @param rows Number of rows in a 2D array.
     * @param cols Number of columns in a 2D array.
     * @Param chs Number of channels. Default: 1.
     */
    _Matrix(int rows, int cols, int chs = 1);

    /**
     * @overload
     * @param size 2D array size: Size(cols, rows).
     * @Param chs Number of channels. Default: 1.
     */
    explicit _Matrix(Size size, int chs = 1);

    /**
     * @overload
     * @code Matrix m(MatrixShape(2, 3, 3)); // Matrix m(2, 3, 3);
     */
    explicit _Matrix(const MatrixShape& shape);

    /**
     * @overload
     * @Param v An optional value to initialize each matrix element with.
     */
    template<typename _T2> _Matrix(int rows, int cols, int chs, const _T2& v);
    template<typename _T2> _Matrix(Size size, int chs,          const _T2& v);
    template<typename _T2> _Matrix(const MatrixShape& shape,    const _T2& v);

    /**
     * @overload
     * @brief Initialize by random number.
     * @code
     *      default_random_engine random_engine(time(nullptr));
     *      uniform_real_distribution<double> real_distribution(-1.0, 1.0);
     *
     *      auto m = _Matrix<double>(4, 4, 1, std::make_pair(random_engine, real_distribution));
     */
    template<typename _T2, typename _T3> _Matrix(int rows, int cols, int chs,   const std::pair<_T2, _T3>& initor);
    template<typename _T2, typename _T3> _Matrix(Size size, int chs,            const std::pair<_T2, _T3>& initor);
    template<typename _T2, typename _T3> _Matrix(const MatrixShape& shape,      const std::pair<_T2, _T3>& initor);

    /**
     * @overload
     * @param m Array that (as a whole or partly) is assigned to the constructed matrix. No data is copied
     * by these constructors. If you want to have an independent copy of the sub-array, use zMatrix::clone() .
     */
    _Matrix(const _Matrix& m);

    /**
     * @brief assignment operator
     * @param m Assigned, right-hand-side matrix.
     */
    _Matrix<_Tp>& operator= (const _Matrix& m);

    /** @overload */
    _Matrix(std::initializer_list<_Tp> list);

    /** @overload */
    _Matrix<_Tp>& operator= (std::initializer_list<_Tp> list);

    /**
     * @brief Cast ctor.
     */
    template<typename _T2> _Matrix(const _Matrix<_T2>& m);
    template<typename _T2> _Matrix<_Tp>& operator= (const _Matrix<_T2>& m);

    ~_Matrix();

    /**
     * @brief Set each matrix element with 'value'.
     */
    template<typename _T2> void fill(const _T2& value);
    void fill(const Scalar& value);
    template<typename _T2> void fill(const _Matrix<_T2>& value);

    /**
     * @brief Set ROI.
     */
    _Matrix(const _Matrix<_Tp>& m, const Rect& roi);

    /**
     *  @brief Creates a full copy of the array and the underlying data.
     */
    _Matrix<_Tp> clone() const;

    /**
     * @brief Copies the matrix to another one.
     */
    void copyTo(_Matrix<_Tp> & outputMatrix) const;

    _Matrix<_Tp> reshape(int cn) const;
    _Matrix<_Tp> reshape(int cn, int rows) const;

    /**
     *  @brief Transposes a matrix.
     */
    _Matrix<_Tp> t();

    /**
     * @brief Trace of the matrix.
     */
    Scalar trace() const;

    /**
     * @brief Inverses a matrix.
     */
//    _Matrix<_Tp> inv();

    /**
     * @brief Performs an element-wise multiplication or division of the two matrices.
     */
    _Matrix<_Tp> mul(const _Matrix<_Tp>& m, double scale=1) const;

    _Matrix<_Matrix<_Tp>> mul(const _Matrix<_Matrix<_Tp>>& m, double scale = 1) const;

    /**
     * @brief Computes a dot-product of two vectors.
     */
    double dot(const _Matrix<_Tp> &m);

    /**
     * @brief Computes a cross-product of two 3-element vectors.
     * The method computes a cross-product of two 3-element vectors. The vectors must be 3-element
     * floating-point vectors of the same shape and size. The result is another 3-element vector of the
     * same shape and type as operands.
     * @param m Another cross-product operand.
     */
    _Matrix<_Tp> cross(_Matrix<_Tp> &m);

    /**
     * @brief Returns a zero array of the specified size and type.
     * @Code
     *      zMatrix m = zMatrix::zeros(10, 10, 3);
     * @Param _rows Number of rows.
     * @Param _cols Number of columns.
     * @Param _chs Number of channels.
     */
    static _Matrix<_Tp> zeros(int _rows, int _cols, int _chs = 1);
    static _Matrix<_Tp> zeros(Size size, int _chs = 1);
    static _Matrix<_Tp> zeros(const MatrixShape& shape);

    /**
     * @brief Returns an array of all 1's of the specified size and type.
     */
    static _Matrix<_Tp> ones(int _rows, int _cols, int _chs = 1);
    static _Matrix<_Tp> ones(Size size, int _chs = 1);
    static _Matrix<_Tp> ones(const MatrixShape& shape);

    /**
     * @brief Returns an identity matrix of the specified size and type.
     */
    static _Matrix<_Tp> eye(int _rows, int _cols, int _chs = 1);
    static _Matrix<_Tp> eye(Size size, int _chs = 1);
    static _Matrix<_Tp> eye(const MatrixShape& shape);

    /**
     * @brief allocates new matrix data unless the matrix already has specified size and type.
     * previous data is unreferenced if needed.
     */
    void create(int rows, int cols, int chs = 1);
    void create(Size size, int chs = 1);
    void create(const MatrixShape& shape);

    /**
     * @brief Returns the number of matrix channels.
     */
    int channels() const;

    /**
     * @overload
     * @param roi Extracted submatrix specified as a rectangle.
     */
    _Matrix<_Tp> operator()(const Rect& roi) const;

    /**
     * @brief Reports whether the matrix is continuous or not.
     */
    bool isContinuous() const;

    /**
     * @brief Return the type of the Matrix.
     */
    int type() const;

    /**
     * @brief Return the depth of the Matrix.
     */
    int depth() const;

    /**
     * @brief Returns true if the array has no elements.
     * The method returns true if the data == nullptr.
     */
    bool empty() const { return !data || !total(); }

    /**
     * @brief Returns the total number of array elements.
     */
    size_t total() const { return size_; }
    size_t size() const { return size_; }

    MatrixShape shape() const { return { rows, cols, channels() }; }

    /**
     * @brief Returns the size of the array element.
     */
    size_t esize() const { return esize_; }

    /**
     * @brief Returns a pointer to the specified matrix row.
     * @param row A 0-based row index.
     */
    _Tp*        ptr(int row = 0);
    const _Tp*  ptr(int row = 0) const;

    /**
     * @overload
     * @brief Return a pointer to the Element.
     * @param row Index along the dimension 0
     * @param col Index along the dimension 1
     */
    _Tp*        ptr(int row, int col);
    const _Tp*  ptr(int row, int col) const;

    /**
     * @overload
     * @brief Return a pointer to the Element.
     * @param row Index along the dimension 0
     * @param col Index along the dimension 1
     * @param ch Index along the dimension 2, channel.
     */
    _Tp*        ptr(int row, int col, int ch);
    const _Tp*  ptr(int row, int col, int ch) const;

    template<typename _T2> _T2*         ptr(int row = 0);
    template<typename _T2> const _T2*   ptr(int row = 0) const;

    template<typename _T2> _T2*         ptr(int row, int col);
    template<typename _T2> const _T2*   ptr(int row, int col) const;

    template<typename _T2> _T2*         ptr(int row, int col, int ch);
    template<typename _T2> const _T2*   ptr(int row, int col, int ch) const;


    /**
     * @brief Returns a reference to the specified array element.
     * @param row Index along the dimension 0
     */
    _Tp&        at(int row = 0);
    const _Tp&  at(int row = 0) const;

    /**
     * @overload
     * @param row Index along the dimension 0
     * @param col Index along the dimension 1
     */
    _Tp&        at(int row, int col);
    const _Tp&  at(int row, int col) const;

    /**
     * @overload
     * @param row Index along the dimension 0
     * @param col Index along the dimension 1
     * @param ch Index along the dimension 2
     */
    _Tp&        at(int row, int col, int ch);
    const _Tp&  at(int row, int col, int ch) const;

    template<typename _T2> _T2&         at(int row = 0);
    template<typename _T2> const _T2&   at(int row = 0) const;

    template<typename _T2> _T2&         at(int row, int col);
    template<typename _T2> const _T2&   at(int row, int col) const;

    template<typename _T2> _T2&         at(int row, int col, int ch);
    template<typename _T2> const _T2&   at(int row, int col, int ch) const;

    _Tp&        operator()(int row = 0);
    const _Tp&  operator()(int row = 0) const;
    _Tp&        operator()(int row, int col);
    const _Tp&  operator()(int row, int col) const;
    _Tp&        operator()(int row, int col, int ch);
    const _Tp&  operator()(int row, int col, int ch) const;

    /**
     * @brief Returns the matrix iterator and sets it to the first matrix element.
     */
    _MatrixIterator<_Tp> begin();
    _MatrixConstIterator<_Tp> begin() const;

    template<typename _T2> _MatrixIterator<_T2> begin();
    template<typename _T2> _MatrixConstIterator<_T2> begin() const;

    /**
     * @brief Returns the matrix iterator and sets it to the after-last matrix element.
     */
    _MatrixIterator<_Tp> end();
    _MatrixConstIterator<_Tp> end() const;

    template<typename _T2> _MatrixIterator<_T2> end();
    template<typename _T2> _MatrixConstIterator<_T2> end() const;


    template<typename _Tp2, class Func> void forEach(const Func& callback);
    template<class Func> void forEach(const Func& callback);

    /**
     * @brief
     */
    _Tp* operator[](size_t n);
    const _Tp* operator[](size_t n) const;


#if defined(USE_OPENCV)
    explicit operator cv::Mat() const;
#endif

    void swap(int32_t i0, int32_t j0, int32_t i1, int32_t j1);

    // Number of rows.
    int rows = 0;
    // Number of columns.
    int cols = 0;

    // Data area
    uint8_t *data = nullptr;

    // Length of rows.
    int step = 0;

private:
    int refAdd(int *addr, int delta);
    void release();

    /**
     * - continuity flag: 1 bits
     * - type: 8 bits
     * - number of channels: 8 bits
     */
    int flags = 0;

    // The start address of the Matrix's data area.
    uint8_t *datastart = nullptr;
    // The end address of the Matrix's data area.
    uint8_t *dataend = nullptr;

    // The size of element.
    size_t esize_ = 0;

    // Number of elements.
    size_t size_ = 0;

    //! pointer to the reference counter;
    // when matrix points to user-allocated data, the pointer is NULL
    int* refcount = nullptr;
};

typedef _Matrix<double>             Matrix64f;
typedef _Matrix<float>              Matrix32f;
typedef _Matrix<signed int>         Matrix32s;
typedef _Matrix<unsigned int>       Matrix32u;
typedef _Matrix<signed short>       Matrix16s;
typedef _Matrix<unsigned short>     Matrix16u;
typedef _Matrix<signed char>        Matrix8s;
typedef _Matrix<unsigned char>      Matrix8u;
typedef _Matrix<unsigned char>      Matrix;

template <typename _Tp> struct Type<_Matrix<_Tp>> { enum { value = TYPE_MATRIX }; };

template <class _Tp> _Matrix<_Tp> operator+(const _Matrix<_Tp> &m1, const _Matrix<_Tp> &m2);
template <class _Tp> _Matrix<_Tp> operator+(const _Matrix<_Tp> &m, const Scalar& delta);
template <class _Tp> _Matrix<_Tp> operator+(const Scalar& delta, const _Matrix<_Tp> &m);
template <class _Tp> _Matrix<_Tp> operator+(const _Matrix<_Tp> &m, double delta);
template <class _Tp> _Matrix<_Tp> operator+(double delta, const _Matrix<_Tp> &m);

template <class _Tp> _Matrix<_Tp> operator+=(_Matrix<_Tp> &m1, const _Matrix<_Tp> &m2);
template <class _Tp> _Matrix<_Tp> operator+=(_Matrix<_Tp> &m, const Scalar& delta);
template <class _Tp> _Matrix<_Tp> operator+=(_Matrix<_Tp> &m, double delta);

template <class _Tp> _Matrix<_Tp> operator-(const _Matrix<_Tp> &m1, const _Matrix<_Tp> &m2);
template <class _Tp> _Matrix<_Tp> operator-(const _Matrix<_Tp> &m, const Scalar& delta);
template <class _Tp> _Matrix<_Tp> operator-(const Scalar& delta, const _Matrix<_Tp> &m);
template <class _Tp> _Matrix<_Tp> operator-(const _Matrix<_Tp> &m, double delta);
template <class _Tp> _Matrix<_Tp> operator-(double delta, const _Matrix<_Tp> &m);

template <class _Tp> _Matrix<_Tp> operator-=(_Matrix<_Tp> &m1, const _Matrix<_Tp> &m2);
template <class _Tp> _Matrix<_Tp> operator-=(_Matrix<_Tp> &m, const Scalar& delta);
template <class _Tp> _Matrix<_Tp> operator-=(_Matrix<_Tp> &m, double delta);

template <class _Tp> _Matrix<_Tp> operator*(const _Matrix<_Tp> &m1, const _Matrix<_Tp> &m2);
template <class _Tp> _Matrix<_Matrix<_Tp>> operator*(const _Matrix<_Tp> &m1, const _Matrix<_Matrix<_Tp>> &m2);
template <class _Tp> _Matrix<_Matrix<_Tp>> operator*(const _Matrix<_Matrix<_Tp>> &m1, const _Matrix<_Tp> &m2);
template <class _Tp> _Matrix<_Tp> operator*(const _Matrix<_Tp> &m, const Scalar& v);
template <class _Tp> _Matrix<_Tp> operator*(const Scalar& v, const _Matrix<_Tp> &m);
template <class _Tp> _Matrix<_Tp> operator*(const _Matrix<_Tp> &m, double v);
template <class _Tp> _Matrix<_Tp> operator*(double v, const _Matrix<_Tp> &m);

template <class _Tp> _Matrix<_Tp> operator*=(_Matrix<_Tp> &m1, const _Matrix<_Tp> &m2);
template <class _Tp> _Matrix<_Tp> operator*=(_Matrix<_Tp> &m, const Scalar& v);
template <class _Tp> _Matrix<_Tp> operator*=(_Matrix<_Tp> &m, double v);

template <class _Tp> _Matrix<_Tp> operator/(const _Matrix<_Tp> &m1, const _Matrix<_Tp> &m2);
template <class _Tp> _Matrix<_Tp> operator/(const _Matrix<_Tp> &m, const Scalar& v);
template <class _Tp> _Matrix<_Tp> operator/(const Scalar& v, const _Matrix<_Tp> &m);
template <class _Tp> _Matrix<_Tp> operator/(const _Matrix<_Tp> &m, double v);
template <class _Tp> _Matrix<_Tp> operator/(double v, const _Matrix<_Tp> &m);

template <class _Tp> _Matrix<_Tp> operator/=(_Matrix<_Tp> &m1, const _Matrix<_Tp> &m2);
template <class _Tp> _Matrix<_Tp> operator/=(_Matrix<_Tp> &m, const Scalar& v);
template <class _Tp> _Matrix<_Tp> operator/=(_Matrix<_Tp> &m, double v);

template <class _Tp> _Matrix<_Tp> operator>(const _Matrix<_Tp> &m1, const _Matrix<_Tp> &m2);
template <class _Tp> _Matrix<_Tp> operator>(const _Matrix<_Tp> &m, const Scalar& threshold);
template <class _Tp> _Matrix<_Tp> operator>(const Scalar& threshold, const _Matrix<_Tp> &m);
template <class _Tp> _Matrix<_Tp> operator>(const _Matrix<_Tp> &m, double threshold);
template <class _Tp> _Matrix<_Tp> operator>(double threshold, const _Matrix<_Tp> &m);

template <class _Tp> _Matrix<_Tp> operator>=(const _Matrix<_Tp> &m1, const _Matrix<_Tp>& m2);
template <class _Tp> _Matrix<_Tp> operator>=(const _Matrix<_Tp> &m, const Scalar& threshold);
template <class _Tp> _Matrix<_Tp> operator>=(const Scalar& threshold, const _Matrix<_Tp> &m);
template <class _Tp> _Matrix<_Tp> operator>=(const _Matrix<_Tp> &m, double threshold);
template <class _Tp> _Matrix<_Tp> operator>=(double threshold, const _Matrix<_Tp> &m);

template <class _Tp> _Matrix<_Tp> operator<(const _Matrix<_Tp> &m1, const _Matrix<_Tp> &m2);
template <class _Tp> _Matrix<_Tp> operator<(const _Matrix<_Tp> &m, const Scalar& threshold);
template <class _Tp> _Matrix<_Tp> operator<(const Scalar& threshold, const _Matrix<_Tp> &m);
template <class _Tp> _Matrix<_Tp> operator<(const _Matrix<_Tp> &m, double threshold);
template <class _Tp> _Matrix<_Tp> operator<(double threshold, const _Matrix<_Tp> &m);

template <class _Tp> _Matrix<_Tp> operator<=(const _Matrix<_Tp> &m1, const _Matrix<_Tp>& m2);
template <class _Tp> _Matrix<_Tp> operator<=(const _Matrix<_Tp> &m, const Scalar& threshold);
template <class _Tp> _Matrix<_Tp> operator<=(const Scalar& threshold, const _Matrix<_Tp> &m);
template <class _Tp> _Matrix<_Tp> operator<=(const _Matrix<_Tp> &m, double threshold);
template <class _Tp> _Matrix<_Tp> operator<=(double threshold, const _Matrix<_Tp> &m);

template <class _Tp> _Matrix<_Tp> operator==(const _Matrix<_Tp> &m1, const _Matrix<_Tp> &m2);
template <class _Tp> _Matrix<_Tp> operator==(const _Matrix<_Tp> &m1, const Scalar& s);
template <class _Tp> _Matrix<_Tp> operator==(const Scalar& s, const _Matrix<_Tp>& m);
template <class _Tp> _Matrix<_Tp> operator==(const _Matrix<_Tp> &m1, double val);
template <class _Tp> _Matrix<_Tp> operator==(double val, const _Matrix<_Tp> &m1);

template <class _Tp> _Matrix<_Tp> operator!=(const _Matrix<_Tp> &m1, const _Matrix<_Tp> &m2);
template <class _Tp> _Matrix<_Tp> operator!=(const _Matrix<_Tp> &m, const Scalar& s);
template <class _Tp> _Matrix<_Tp> operator!=(const Scalar& s, const _Matrix<_Tp> &m);
template <class _Tp> _Matrix<_Tp> operator!=(const _Matrix<_Tp> &m1, double val);
template <class _Tp> _Matrix<_Tp> operator!=(double val, const _Matrix<_Tp> &m2);

template <class _Tp> std::ostream &operator<<(std::ostream & os, const _Matrix<_Tp> &item);

///////////////////////////////////////// operations ////////////////////////////////////////////
template<typename _Tp, class Functor> void traverse(_Matrix<_Tp>& m1, std::ptrdiff_t diff, const Functor& callback);
template<typename _Tp, class Functor> void traverse(const _Matrix<_Tp>& m1, std::ptrdiff_t diff, const Functor& callback);

template<typename _Tp, class Functor> void traverse(_Matrix<_Tp>& m1, _Matrix<_Tp>& m2, std::ptrdiff_t diff, const Functor& callback);
template<typename _Tp, class Functor> void traverse(const _Matrix<_Tp>& m1, _Matrix<_Tp>& m2, std::ptrdiff_t diff, const Functor& callback);
template<typename _Tp, typename _T2, class Functor> void traverse(_Matrix<_Tp>& m1, _Matrix<_T2>& m2, std::ptrdiff_t diff, const Functor& callback);
template<typename _Tp, typename _T2, class Functor> void traverse(const _Matrix<_Tp>& m1, _Matrix<_T2>& m2, std::ptrdiff_t diff, const Functor& callback);

template<typename _Tp, class Functor> void traverse(_Matrix<_Tp>& m1, _Matrix<_Tp>& m2, _Matrix<_Tp>& m3, std::ptrdiff_t diff, const Functor& callback);
template<typename _Tp, class Functor> void traverse(const _Matrix<_Tp>& m1, const _Matrix<_Tp>& m2, _Matrix<_Tp>& m3, std::ptrdiff_t diff, const Functor& callback);
template<typename _Tp, typename _T2, typename _T3, class Functor> void traverse(_Matrix<_Tp>& m1, _Matrix<_T2>& m2, _Matrix<_T3>& m3, std::ptrdiff_t diff, const Functor& callback);
template<typename _Tp, typename _T2, typename _T3, class Functor> void traverse(const _Matrix<_Tp>& m1, const _Matrix<_T2>& m2, _Matrix<_T3>& m3, std::ptrdiff_t diff, const Functor& callback);

template<typename _Tp, class Functor> void traverse(_Matrix<_Tp>& m1, _Matrix<_Tp>& m2, _Matrix<_Tp>& m3, _Matrix<_Tp>& m4, std::ptrdiff_t diff, const Functor& callback);
template<typename _Tp, class Functor> void traverse(const _Matrix<_Tp>& m1, const _Matrix<_Tp>& m2, const _Matrix<_Tp>& m3, _Matrix<_Tp>& m4, std::ptrdiff_t diff, const Functor& callback);

template<typename _Tp> Scalar trace(const _Matrix<_Tp>& m);

template<typename _Tp> _Matrix<_Tp> abs(const _Matrix<_Tp>& m);

template<typename _Tp> void absdiff(const _Matrix<_Tp>& m1, const _Matrix<_Tp>& m2, _Matrix<_Tp>& dst);
template<typename _Tp> void absdiff(const Scalar& value, const _Matrix<_Tp>& m2, _Matrix<_Tp>& dst);
template<typename _Tp> void absdiff(const _Matrix<_Tp>& m1, const Scalar& value, _Matrix<_Tp>& dst);

template<typename _Tp> void add(const _Matrix<_Tp>& m1, const _Matrix<_Tp>& m2, _Matrix<_Tp>& dst, const Matrix& mask=Matrix());

template <typename _Tp> void addWeighted(const _Matrix<_Tp>&src1, double alpha, const _Matrix<_Tp>&src2, double beta, double gamma, _Matrix<_Tp>&dst);

/////////////////////////////////////////_MatrixConstIterator////////////////////////////////////////////
template<typename _Tp> class _MatrixConstIterator {
public:
    typedef _Tp value_type;
    typedef std::ptrdiff_t difference_type;
    typedef const _Tp* pointer;
    typedef const _Tp& reference;

    typedef std::random_access_iterator_tag iterator_category;

    _MatrixConstIterator() = default;
    _MatrixConstIterator(const _Matrix<_Tp>* m);
    _MatrixConstIterator(const _MatrixConstIterator<_Tp>& it);

    _MatrixConstIterator<_Tp>& operator=(const _MatrixConstIterator<_Tp>& it);

    const _Tp& operator *() const;
    const _Tp& operator[](difference_type i) const;

    _MatrixConstIterator& operator+=(difference_type ofs);
    _MatrixConstIterator& operator-=(difference_type ofs);

    _MatrixConstIterator& operator--();
    _MatrixConstIterator  operator--(int);
    _MatrixConstIterator& operator++();
    _MatrixConstIterator  operator++(int);

    bool operator == (const _MatrixConstIterator<_Tp>& it) const;
    bool operator != (const _MatrixConstIterator<_Tp>& it) const;

    bool operator < (const _MatrixConstIterator<_Tp>& it) const;
    bool operator > (const _MatrixConstIterator<_Tp>& it) const;

    void seek(difference_type ofs, bool relative = false);

protected:
    const _Matrix<_Tp>* m_ = nullptr;
    size_t esize_ = 0;
    const uint8_t* ptr_ = nullptr;
    const uint8_t* start_ = nullptr;
    const uint8_t* end_ = nullptr;
};

template <typename _Tp>
class _MatrixIterator : public _MatrixConstIterator<_Tp> {
public:
    typedef _Tp value_type;
    typedef std::ptrdiff_t difference_type;
    typedef _Tp* pointer;
    typedef _Tp& reference;

    typedef std::random_access_iterator_tag iterator_category;

    _MatrixIterator() = default;
    _MatrixIterator(_Matrix<_Tp>* _m);
    _MatrixIterator(const _MatrixIterator& it);
    _MatrixIterator& operator = (const _MatrixIterator<_Tp>& it);

    _Tp& operator *() const;
    _Tp& operator [](difference_type i) const;

    _MatrixIterator& operator += (difference_type ofs);
    _MatrixIterator& operator -= (difference_type ofs);

    _MatrixIterator& operator --();
    _MatrixIterator  operator --(int);
    _MatrixIterator& operator ++();
    _MatrixIterator  operator ++(int);
};
} //! namespace z

#include "matrix.hpp"

#endif  //! _ZCORE_MATRIX_H