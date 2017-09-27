#ifndef _ZCORE_TRAITS_HPP
#define _ZCORE_TRAITS_HPP

#include <stdint.h>

namespace z {
//////////////////////////////////////////////////////////////////////////////////////////
template <typename T> class MatrixType {
public:
    enum {
        depth = -1,
        channels = 1,
        type = depth << 8 | channels
    };
};

// ÌØ»¯
template <> class MatrixType<uint8_t> { public: enum { depth = 0, channels = 1, type = depth << 8 | channels }; };
template <> class MatrixType<int8_t> { public: enum { depth = 1, channels = 1, type = depth << 8 | channels }; };
template <> class MatrixType<uint16_t> { public: enum { depth = 2, channels = 1, type = depth << 8 | channels }; };
template <> class MatrixType<int16_t> { public: enum { depth = 3, channels = 1, type = depth << 8 | channels }; };
template <> class MatrixType<int32_t> { public: enum { depth = 4, channels = 1, type = depth << 8 | channels }; };
template <> class MatrixType<float> { public: enum { depth = 5, channels = 1, type = depth << 8 | channels }; };
template <> class MatrixType<double> { public: enum { depth = 6, channels = 1, type = depth << 8 | channels }; };

};
#endif // !_ZCORE_TRAITS_HPP