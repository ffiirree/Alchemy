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

    static void bernoulli_fill(int count, T* ptr, double probability);
    static void uniform_fill(int count, T * ptr, double a, double b);
    static void normal_fill(int count, T * ptr, double mean, double stddev);

private:

    static void xavier_fill(int count, T *ptr, int N);
};

}



#endif //! ALCHEMY_UTIL_FILLER_H
