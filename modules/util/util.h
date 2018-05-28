#ifndef ALCHEMY_UTIL_UTIL_H
#define ALCHEMY_UTIL_UTIL_H

#define TO_STRING(x)    #x
#define st(x)           do { x } while (__LINE__ == -1)
#define __unused_parameter__(param)   {(param) = (param);}

namespace alchemy {
struct CPU {};

struct GPU {};

struct XPU {};
}

#endif  // !ALCHEMY_UTIL_UTIL_H