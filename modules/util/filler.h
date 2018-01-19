#ifndef ALCHEMY_UTIL_FILLER_H
#define ALCHEMY_UTIL_FILLER_H

#include <core/tensor.h>

namespace alchemy {

enum FillerType {
    UNIFORM,
    NORMAL,
    XAVIER,
    BERNOULLI
};

template <typename T>
class Filler{
public:

    static void fill(Tensor<T>& tensor, FillerType type);
    static void fill(Tensor<T>& tensor, FillerType type, double probability);

private:
    static void uniform_fill(const int count, T * ptr);
    static void normal_fill(const int count, T * ptr);
    static void xavier_fill(const int count, const T scale, T *ptr);
};

}



#endif //! ALCHEMY_UTIL_FILLER_H
