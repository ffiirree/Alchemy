#ifndef __ZMATH_ZMATH_HPP
#define __ZMATH_ZMATH_HPP

namespace z {
template <typename _Tp> void max(_Tp *addr, size_t size, _Tp & _max)
{
    _max = *addr;
    _Tp *begin = addr + 1;
    _Tp *end = addr + size;

    for (; begin < end; ++begin) {
        if (_max < begin[0])
            _max = begin[0];
    }
}



template <typename _Tp> void min(_Tp *addr, size_t size, _Tp & _min)
{
    _min = *addr;
    _Tp *begin = addr + 1;
    _Tp *end = addr + size;

    for (; begin < end; ++begin) {
        if (_min > begin[0])
            _min = begin[0];
    }
}
}

#endif // !__ZMATH_ZMATH_HPP