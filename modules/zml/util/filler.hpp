#ifndef _ZML_FILLER_HPP
#define _ZML_FILLER_HPP

#include <zml/tensor.hpp>

namespace z {

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



#endif //! _ZML_FILLER_HPP
