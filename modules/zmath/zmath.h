/**
 ******************************************************************************
 * @file    zmath.cpp
 * @author  zlq
 * @version V1.0
 * @date    2016.9.14
 * @brief   简单的数学函数定义
 ******************************************************************************
 * @attention
 *
 *
 ******************************************************************************
 */
#ifndef _ZMATH_ZMATH_H
#define _ZMATH_ZMATH_H


#define Pi			((double)3.14159265358979323846)

#define fequ(temp)      (fabs(temp) < 10e-5)
#define dequ(temp)      (fabs(temp) < 10e-6)

namespace z {
template <typename _Tp> void min(_Tp *addr, size_t size, _Tp & _min);
template <typename _Tp> void max(_Tp *addr, size_t size, _Tp & _max);

int average(int x, int y);

};

/**
* @brief 求数组中的最大数
* @param size 数组的大小 or 需要比较数组前size个
*/
template <typename _Tp> void z::max(_Tp *addr, size_t size, _Tp & _max)
{
    _max = *addr;
    _Tp *begin = addr + 1;
    _Tp *end = addr + size;

    for (; begin < end; ++begin) {
        if (_max < begin[0])
            _max = begin[0];
    }
}


/**
* @brief 求数组中的最大数
* @param size 数组的大小 or 需要比较数组前size个
*/
template <typename _Tp> void z::min(_Tp *addr, size_t size, _Tp & _min)
{
    _min = *addr;
    _Tp *begin = addr + 1;
    _Tp *end = addr + size;

    for (; begin < end; ++begin) {
        if (_min > begin[0])
            _min = begin[0];
    }
}



#endif // !_ZMATH_ZMATH_H
