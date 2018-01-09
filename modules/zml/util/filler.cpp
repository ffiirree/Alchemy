#include <glog/logging.h>
#include <random>
#include "filler.hpp"

namespace z {

template<typename T>
void Filler<T>::fill(Tensor<T>& tensor, FillerType type)
{
    const auto count = tensor.count();
    const auto ptr = tensor.data();
    const auto num = tensor.shape(0);

    switch(type) {
        case NORMAL:
            normal_fill(count, ptr);
            break;

        case UNIFORM:
            uniform_fill(count, ptr);
            break;

        case XAVIER:
            xavier_fill(count, (T)1./((T)count/ num), ptr);
            break;

        default:
            LOG(FATAL) << "Unknown filler type!";
            break;
    }
}

template<typename T>
void Filler<T>::uniform_fill(const int count, T *ptr)
{
    std::default_random_engine random_engine(static_cast<unsigned long>(time(nullptr)));
    std::uniform_real_distribution<double> uniform_distribution(-1.0, 1.0);

    for(auto i = 0; i < count; ++i) {
        ptr[i] = uniform_distribution(random_engine);
    }
}

template<typename T>
void Filler<T>::normal_fill(const int count, T *ptr)
{
    std::default_random_engine random_engine(static_cast<unsigned long>(time(nullptr)));
    std::normal_distribution<double> normal_distribution(0, 1.0);

    for(auto i = 0; i < count; ++i) {
        ptr[i] = normal_distribution(random_engine);
    }
}


template<typename T>
void Filler<T>::xavier_fill(const int count, const T scale, T *ptr)
{
    std::default_random_engine random_engine(static_cast<unsigned long>(time(nullptr)));
    std::normal_distribution<double> normal_distribution(0, 1.0);

    for(auto i = 0; i < count; ++i) {
        ptr[i] = normal_distribution(random_engine) * scale;
    }
}

template class Filler<float>;
template class Filler<double>;

}