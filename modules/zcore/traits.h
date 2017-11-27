#ifndef _ZCORE_TRAITS_HPP
#define _ZCORE_TRAITS_HPP

#include <cstdint>

namespace z {
//////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
struct DataType {
    enum {
        depth = -1,
        channels = 1,
        value = depth << 8 | channels
    };
};

// 特化
template <> struct DataType<uint8_t>   { enum { depth = 0, channels = 1, value = depth << 8 | channels }; };
template <> struct DataType<int8_t>    { enum { depth = 1, channels = 1, value = depth << 8 | channels }; };
template <> struct DataType<char>      { enum { depth = 1, channels = 1, value = depth << 8 | channels }; };
template <> struct DataType<uint16_t>  { enum { depth = 2, channels = 1, value = depth << 8 | channels }; };
template <> struct DataType<int16_t>   { enum { depth = 3, channels = 1, value = depth << 8 | channels }; };
template <> struct DataType<int32_t>   { enum { depth = 4, channels = 1, value = depth << 8 | channels }; };
template <> struct DataType<float>     { enum { depth = 5, channels = 1, value = depth << 8 | channels }; };
template <> struct DataType<double>    { enum { depth = 6, channels = 1, value = depth << 8 | channels }; };

};
#endif // !_ZCORE_TRAITS_HPP