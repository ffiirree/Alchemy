#ifndef ALCHEMY_UTIL_SATURATE_H
#define ALCHEMY_UTIL_SATURATE_H

#include <algorithm>

namespace alchemy {
//////////////////////////////////////////////////////////////////////////
template<typename _T1, typename _T2> static _T1 saturate_cast(_T2 v) { return _T1(v); }

template<typename _Tp> static _Tp saturate_cast(uint8_t v)  { return _Tp(v); }
template<typename _Tp> static _Tp saturate_cast(int8_t v)   { return _Tp(v); }
template<typename _Tp> static _Tp saturate_cast(uint16_t v) { return _Tp(v); }
template<typename _Tp> static _Tp saturate_cast(int16_t v)  { return _Tp(v); }
template<typename _Tp> static _Tp saturate_cast(uint32_t v) { return _Tp(v); }
template<typename _Tp> static _Tp saturate_cast(int32_t v)  { return _Tp(v); }
template<typename _Tp> static _Tp saturate_cast(float v)    { return _Tp(v); }
template<typename _Tp> static _Tp saturate_cast(double v)   { return _Tp(v); }
template<typename _Tp> static _Tp saturate_cast(int64_t v)  { return _Tp(v); }
template<typename _Tp> static _Tp saturate_cast(uint64_t v) { return _Tp(v); }

template <> inline uint8_t saturate_cast<uint8_t>(int8_t v)     { return static_cast<uint8_t>(std::max(static_cast<int>(v), 0)); }
template <> inline uint8_t saturate_cast<uint8_t>(uint16_t v)   { return static_cast<uint8_t>(std::min(static_cast<unsigned>(v), static_cast<unsigned>(UINT8_MAX))); }
template <> inline uint8_t saturate_cast<uint8_t>(int32_t v)    { return static_cast<uint8_t>(static_cast<unsigned>(v) <= UINT8_MAX ? v : v > 0 ? UINT8_MAX : 0); }
template <> inline uint8_t saturate_cast<uint8_t>(int16_t v)    { return saturate_cast<uint8_t>(int(v)); }
template <> inline uint8_t saturate_cast<uint8_t>(uint32_t v)   { return static_cast<uint8_t>(std::min(v, static_cast<unsigned>(UINT8_MAX))); }
template <> inline uint8_t saturate_cast<uint8_t>(float v)      { return saturate_cast<uint8_t>(static_cast<int>(v)); }
template <> inline uint8_t saturate_cast<uint8_t>(double v)     { return saturate_cast<uint8_t>(static_cast<int>(v)); }
template <> inline uint8_t saturate_cast<uint8_t>(int64_t v)    { return static_cast<uint8_t>(static_cast<uint64_t>(v) <= static_cast<uint64_t>(UINT8_MAX) ? v : v > 0 ? UINT8_MAX : 0); }
template <> inline uint8_t saturate_cast<uint8_t>(uint64_t v)   { return static_cast<uint8_t>(std::min(v, static_cast<uint64_t>(UINT8_MAX))); }

template <> inline int8_t saturate_cast<int8_t>(int8_t v)   { return static_cast<int8_t>(std::min(v, static_cast<int8_t>(INT8_MAX))); }
template <> inline int8_t saturate_cast<int8_t>(uint16_t v) { return static_cast<int8_t>(std::min(v, static_cast<uint16_t>(INT8_MAX))); }
template <> inline int8_t saturate_cast<int8_t>(int32_t v)  { return static_cast<int8_t>(static_cast<unsigned>(v - INT8_MIN) <= static_cast<unsigned>(UINT8_MAX) ? v : v > 0 ? INT8_MAX : INT8_MIN);}
template <> inline int8_t saturate_cast<int8_t>(int16_t v)  { return saturate_cast<int8_t>(int(v)); }
template <> inline int8_t saturate_cast<int8_t>(uint32_t v) { return static_cast<int8_t>(std::min(v, static_cast<unsigned>(INT8_MAX))); }
template <> inline int8_t saturate_cast<int8_t>(float v)    { return saturate_cast<int8_t>(static_cast<int>(v)); }
template <> inline int8_t saturate_cast<int8_t>(double v)   { return saturate_cast<int8_t>(static_cast<int>(v)); }
template <> inline int8_t saturate_cast<int8_t>(int64_t v)  { return static_cast<int8_t>(static_cast<uint64_t>(static_cast<int64_t>(v) - INT8_MIN) <= static_cast<uint64_t>(UINT8_MAX) ? v : v > 0 ? INT8_MAX : INT8_MIN);}
template <> inline int8_t saturate_cast<int8_t>(uint64_t v) { return static_cast<int8_t>(std::min(v, static_cast<uint64_t>(INT8_MAX))); }

template <> inline uint16_t saturate_cast<uint16_t>(int8_t v)   { return static_cast<uint16_t>(std::max(static_cast<int>(v), 0)); }
template <> inline uint16_t saturate_cast<uint16_t>(int16_t v)  { return static_cast<uint16_t>(std::max(static_cast<int>(v), 0)); }
template <> inline uint16_t saturate_cast<uint16_t>(int32_t v)  { return static_cast<uint16_t>(static_cast<unsigned>(v) <= UINT16_MAX ? v : v > 0 ? UINT16_MAX : 0); }
template <> inline uint16_t saturate_cast<uint16_t>(uint32_t v) { return static_cast<uint16_t>(std::min(v, static_cast<unsigned>(UINT16_MAX))); }
template <> inline uint16_t saturate_cast<uint16_t>(float v)    { return saturate_cast<uint16_t>(static_cast<int>(v)); }
template <> inline uint16_t saturate_cast<uint16_t>(double v)   { return saturate_cast<uint16_t>(static_cast<int>(v)); }
template <> inline uint16_t saturate_cast<uint16_t>(int64_t v)  { return static_cast<uint16_t>(static_cast<uint64_t>(v) <= static_cast<uint64_t>(UINT16_MAX) ? v : v > 0 ? UINT16_MAX : 0); }
template <> inline uint16_t saturate_cast<uint16_t>(uint64_t v) { return static_cast<uint16_t>(std::min(v, static_cast<uint64_t>(UINT16_MAX))); }

template<> inline int16_t saturate_cast<int16_t>(uint16_t v)  { return static_cast<short>(std::min(static_cast<int>(v), static_cast<int>(INT16_MAX))); }
template<> inline int16_t saturate_cast<int16_t>(int v)       { return static_cast<int16_t>(static_cast<unsigned>(v - INT16_MIN) <= static_cast<unsigned>(UINT16_MAX) ? v : (v > 0 ? INT16_MAX : INT16_MIN)); }
template<> inline int16_t saturate_cast<int16_t>(unsigned v)  { return static_cast<int16_t>(std::min(v, static_cast<unsigned>(INT16_MAX))); }
template<> inline int16_t saturate_cast<int16_t>(float v)     { return saturate_cast<int16_t>(static_cast<int>(v)); }
template<> inline int16_t saturate_cast<int16_t>(double v)    { return saturate_cast<int16_t>(static_cast<int>(v)); }
template<> inline int16_t saturate_cast<int16_t>(int64_t v)   { return static_cast<int16_t>(static_cast<uint64_t>(static_cast<int64_t>(v) - INT16_MIN) <= static_cast<uint64_t>(UINT16_MAX) ? v : v > 0 ? INT16_MAX : INT16_MIN); }
template<> inline int16_t saturate_cast<int16_t>(uint64_t v)  { return static_cast<int16_t>(std::min(v, static_cast<uint64_t>(INT16_MAX))); }

};

#endif //! ALCHEMY_UTIL_SATURATE_H