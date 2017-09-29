#ifndef __ZCORE_MATRIX_COMPUTE_H
#define __ZCORE_MATRIX_COMPUTE_H

namespace z{

template <typename _Tp> class _Matrix;

struct MatrixCompute
{
    template<typename _Tp, class Func> _Matrix<_Tp> operator()(const _Matrix<_Tp>& m1, const _Matrix<_Tp>& m2, Func callback);
    template<typename _Tp, class Func> _Matrix<_Tp> operator()(_Matrix<_Tp>& m1, const _Matrix<_Tp>& m2, Func callback);

    template<typename _Tp, class Func> _Matrix<_Tp> operator()(const _Matrix<_Tp>& m1, double _value, Func callback);
    template<typename _Tp, class Func> _Matrix<_Tp> operator()(_Matrix<_Tp>& m1, double _value, Func callback);

    template<typename _Tp, class Func> _Matrix<_Tp> operator()(const _Matrix<_Tp>& m1, const Scalar& delta, Func callback);
    template<typename _Tp, class Func> _Matrix<_Tp> operator()(_Matrix<_Tp>& m1, const Scalar& delta, Func callback);
};

}

#include "matrix_compute.hpp"

#endif // !__ZCORE_MATRIX_COMPUTE_H