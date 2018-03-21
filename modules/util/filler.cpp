#include "filler.h"
#include <random>

namespace alchemy {

template<typename T>
void Filler<T>::fill(const Tensor<T>& tensor, FillerType type)
{
    const auto count = tensor.count();
    const auto ptr = tensor.cptr();

    switch(type) {
        case NORMAL:
            normal_fill(count, ptr, 0.0, 1.0);
            break;

        case UNIFORM:
            uniform_fill(count, ptr, -1.0, 1.0);
            break;

        case XAVIER:
            xavier_fill(count, ptr, count / tensor.shape(0));
            break;

        case CONSTANT:
            constant_fill(count, ptr, 0);
            break;

        default:
            LOG(FATAL) << "Unknown filler type!";
            break;
    }
}

template<typename T>
void Filler<T>::uniform_fill(int count, T * ptr, double a, double b)
{
    std::default_random_engine random_engine(static_cast<unsigned long>(time(nullptr)));
    std::uniform_real_distribution<double> uniform_distribution(a, b);

    for(auto i = 0; i < count; ++i) {
        ptr[i] = uniform_distribution(random_engine);
    }
}

template<typename T>
void Filler<T>::normal_fill(int count, T * ptr, double mean, double stddev)
{
    std::default_random_engine random_engine(static_cast<unsigned long>(time(nullptr)));
    std::normal_distribution<double> normal_distribution(mean, stddev);

    for(auto i = 0; i < count; ++i) {
        ptr[i] = normal_distribution(random_engine);
    }
}

template<typename T>
void Filler<T>::bernoulli_fill(int count, T* ptr, double probability)
{
    std::default_random_engine random_engine(static_cast<unsigned long>(time(nullptr)));
    std::bernoulli_distribution bernoulli_distribution(probability);

    for(auto i = 0; i < count; ++i) {
        ptr[i] = bernoulli_distribution(random_engine);
    }
}

template<typename T>
void Filler<T>::xavier_fill(int count, T *ptr, int N)
{
    const double scale = std::sqrt(3.0/N);

    uniform_fill(count, ptr, -scale, scale);
}

template<typename T>
void Filler<T>::constant_fill(int count, T *ptr, const T value)
{
    vector_set(count, value, ptr);
}

template class Filler<uint8_t>;
template class Filler<float>;
template class Filler<double>;
}