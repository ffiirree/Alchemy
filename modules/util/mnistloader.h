#ifndef ALCHEMY_UTIL_MNISTLOADER_H
#define ALCHEMY_UTIL_MNISTLOADER_H

#include <string>
#include <vector>
#include <core/matrix.h>

namespace alchemy
{
class MnistLoader {
public:
    enum {
        LABEL_MAGIC_NUMBER = 2049, IMAGE_MAGIC_NUMBER = 2051
    };

    using Pair = std::pair<Matrix, uint8_t>;

public:
    MnistLoader(const std::string &images_path, const std::string &labels_path);
    MnistLoader(const MnistLoader &) = delete;
    MnistLoader &operator=(const MnistLoader &) = delete;
    ~MnistLoader() = default;

    bool empty() const { return !size_; }
    uint32_t size() const { return size_; }

    std::vector<Pair> data() const { return data_; }

private:
    bool loadImages(const std::string &path);
    bool loadLabels(const std::string &path);

    static uint32_t reverse(uint32_t x);

    uint32_t rows_ = 0;
    uint32_t cols_ = 0;
    uint32_t size_ = 0;

    std::vector<Pair> data_;
}; //! class MnistLoader
} //! namespace z

#endif  //! ALCHEMY_UTIL_MNISTLOADER_H