#ifndef ALCHEMY_GTEST_DEFINES_H
#define ALCHEMY_GTEST_DEFINES_H

namespace alchemy {

template <typename TA, typename TB>
struct TypeDefinitions {
    using TypeA = TA;
    using TypeB = TB;
};
}

#endif //! ALCHEMY_GTEST_DEFINES_H
