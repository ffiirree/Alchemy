/**
 ******************************************************************************
 * @file    zmath.h
 * @author  zlq
 * @version V1.0
 * @date    2016.9.14
 ******************************************************************************
 * @attention
 *
 *
 ******************************************************************************
 */
#ifndef _ZMATH_ZMATH_H
#define _ZMATH_ZMATH_H

#include <cstddef>


#define Pi			((double)3.14159265358979323846)

#define fequ(temp)      (fabs(temp) < 10e-5)
#define dequ(temp)      (fabs(temp) < 10e-6)

namespace z {

/**
 * @brief Find the smallest number in the array
 * @param[in] addr Pointer to the Array.
 * @param[in] size The number of elements in the array.
 * @param[out] _min The smallest element.
 */
template <typename _Tp> void min(_Tp *addr, size_t size, _Tp & _min);

/**
 * @brief Find the largest number in the array
 * @param[in] addr Pointer to the Array.
 * @param[in] size The number of elements in the array.
 * @param[out] _max The largest element.
 */
template <typename _Tp> void max(_Tp *addr, size_t size, _Tp & _max);

};

#include "zmath.hpp"

#endif // !_ZMATH_ZMATH_H
