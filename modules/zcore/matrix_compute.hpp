#ifndef __ZCORE_MATRIX_MatrixCompute_HPP
#define __ZCORE_MATRIX_MatrixCompute_HPP

namespace z{

template <typename _Tp, class Func>
_Matrix<_Tp> MatrixCompute::operator()(const _Matrix<_Tp>& m1, const _Matrix<_Tp>& m2, Func callback)
{
    assert(m1.size() == m2.size());
    assert(m1.channels() == m2.channels());

    auto rm = z::_Matrix<_Tp>(m1.size(), m1.channels());

    auto _len = m1.cols * m1.channels();
    for (auto i = 0; i < m1.rows; ++i) {
        auto _ptr_1 = m1.ptr(i);
        auto _ptr_2 = m2.ptr(i);
        auto _ptr_3 = rm.ptr(i);
        for (auto j = 0; j < _len; ++j) {
            _ptr_3[j] = saturate_cast<_Tp>(callback(_ptr_1[j], _ptr_2[j]));
        }
    }
    return rm;
}

template <typename _Tp, class Func>
_Matrix<_Tp> MatrixCompute::operator()(_Matrix<_Tp>& m1, const _Matrix<_Tp>& m2, Func callback)
{
    assert(m1.size() == m2.size());
    assert(m1.channels() == m2.channels());

    auto _len = m1.cols * m1.channels();
    for (auto i = 0; i < m1.rows; ++i) {
        auto _ptr_1 = m1.ptr(i);
        auto _ptr_2 = m2.ptr(i); \
            for (auto j = 0; j < _len; ++j) {
                _ptr_1[j] = saturate_cast<_Tp>(callback(_ptr_1[j], _ptr_2[j]));
            }
    }
    return m1;
}

template <typename _Tp, class Func>
_Matrix<_Tp> MatrixCompute::operator()(const _Matrix<_Tp>& m, double _value, Func callback)
{
    auto rm = z::_Matrix<_Tp>(m.size(), m.channels());

    auto _len = m.cols * m.channels();
    for (auto i = 0; i < m.rows; ++i) {
        auto _ptr_1 = m.ptr(i);
        auto _ptr_2 = rm.ptr(i);
        for (auto j = 0; j < _len; ++j) {
            _ptr_2[j] = saturate_cast<_Tp>(callback(_ptr_1[j]));
        }
    }
    return rm;
}

template <typename _Tp, class Func>
_Matrix<_Tp> MatrixCompute::operator()(_Matrix<_Tp>& m, double _value, Func callback)
{
    auto _len = m.cols * m.channels();
    for (auto i = 0; i < m.rows; ++i) {
        auto _ptr_1 = m.ptr(i);
        for (auto j = 0; j < _len; ++j) {
            _ptr_1[j] = saturate_cast<_Tp>(callback(_ptr_1[j]));
        }
    }
    return m;
}

template <typename _Tp, class Func>
_Matrix<_Tp> MatrixCompute::operator()(const _Matrix<_Tp>& m, const Scalar& delta, Func callback)
{
    assert(m.channels() <= 4);

    auto rm = z::_Matrix<_Tp>(m.size(), m.channels());

    for (auto i = 0; i < m.rows; ++i) {
        for (auto j = 0; j < m.cols; ++j) {
            for (auto k = 0; k < m.channels(); ++k) {
                rm.at(i, j, k) = saturate_cast<_Tp>(callback(m.at(i, j, k), delta[k]));
            }
        }
    }
    return rm;
}

template <typename _Tp, class Func>
_Matrix<_Tp> MatrixCompute::operator()(_Matrix<_Tp>& m, const Scalar& delta, Func callback)
{
    assert(m.channels() <= 4);

    for (auto i = 0; i < m.rows; ++i) {
        for (auto j = 0; j < m.cols; ++j) {
            for (auto k = 0; k < m.channels(); ++k) {
                auto&& _value = m.at(i, j, k);
                _value = saturate_cast<_Tp>(callback(_value, delta[k]));
            }
        }
    }
    return m;
}
}

#endif // !__ZCORE_MATRIX_MatrixCompute_HPP