#ifndef _ZCORE_TRAITS_HPP
#define _ZCORE_TRAITS_HPP

#include <cstdint>
#include "zdef.h"

namespace z {
enum
{
    DEPTH_DEFAULT   = -1,
    DEPTH_UINT8     = 0,
    DEPTH_INT8      = 1,
    DEPTH_UINT16    = 2,
    DEPTH_INT15     = 3,
    DEPTH_INT32     = 4,
    DEPTH_FLOAT     = 5,
    DEPTH_DOUBLE    = 6,
    DEPTH_UINT32    = 7,
    DEPTH_UINT64    = 8,
    DEPTH_INT64     = 9
};
//////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
struct DataType {
    enum {
        depth = DEPTH_DEFAULT,
        channels = 1,
        value = depth << 8 | channels
    };
};

// 特化
template <> struct DataType<uint8_t>   { enum { depth = DEPTH_UINT8,     channels = 1, value = depth << 8 | channels }; };
template <> struct DataType<int8_t>    { enum { depth = DEPTH_INT8,      channels = 1, value = depth << 8 | channels }; };
template <> struct DataType<char>      { enum { depth = DEPTH_INT8,      channels = 1, value = depth << 8 | channels }; };
template <> struct DataType<uint16_t>  { enum { depth = DEPTH_UINT16,    channels = 1, value = depth << 8 | channels }; };
template <> struct DataType<int16_t>   { enum { depth = DEPTH_INT15,     channels = 1, value = depth << 8 | channels }; };
template <> struct DataType<int32_t>   { enum { depth = DEPTH_INT32,     channels = 1, value = depth << 8 | channels }; };
template <> struct DataType<float>     { enum { depth = DEPTH_FLOAT,     channels = 1, value = depth << 8 | channels }; };
template <> struct DataType<double>    { enum { depth = DEPTH_DOUBLE,    channels = 1, value = depth << 8 | channels }; };

    
//////////////////////////////////////////////////////////////////////////////////////////
template <typename _Tp> struct Type { enum { value = TYPE_DEFAULT };};

template <> struct Type<uint8_t>    { enum { value = TYPE_UINT8 }; };
template <> struct Type<int8_t>     { enum { value = TYPE_INT8 }; };
template <> struct Type<char>       { enum { value = TYPE_INT8 }; };
template <> struct Type<uint16_t>   { enum { value = TYPE_UINT16 }; };
template <> struct Type<int16_t>    { enum { value = TYPE_INT16 }; };
template <> struct Type<uint32_t>   { enum { value = TYPE_UINT32 }; };
template <> struct Type<int32_t>    { enum { value = TYPE_INT32 }; };
template <> struct Type<uint64_t>   { enum { value = TYPE_UINT64 }; };
template <> struct Type<int64_t>    { enum { value = TYPE_INT64 }; };
template <> struct Type<float>      { enum { value = TYPE_FLOAT }; };
template <> struct Type<double>     { enum { value = TYPE_DOUBLE }; };

};
#endif // !_ZCORE_TRAITS_HPP