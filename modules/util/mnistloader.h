#ifndef ALCHEMY_UTIL_MNISTLOADER_H
#define ALCHEMY_UTIL_MNISTLOADER_H

#include <string>
#include <vector>
#include <memory>
#include "core/matrix.h"

namespace alchemy
{
class MnistLoader {
public:
    enum {
        LABEL_MAGIC_NUMBER = 2049, IMAGE_MAGIC_NUMBER = 2051
    };

public:
    MnistLoader() = default;
    MnistLoader(const std::string &images_path, const std::string &labels_path);
    MnistLoader(const MnistLoader &) = default;
    MnistLoader& operator=(const MnistLoader &) = default;
    ~MnistLoader() = default;

    inline bool empty() const { return !size_; }
    inline auto size() const { return size_; }

    inline auto rows() const { return rows_; }
    inline auto cols() const { return cols_; }
    inline auto chs() const { return (uint32_t)1; }

    inline auto images() const { return images_; }
    inline auto labels() const { return labels_; }

    inline auto classification_num() const { return classification_num_; }

private:
    void loadImages(const std::string &path);
    void loadLabels(const std::string &path);

    static uint32_t reverse(uint32_t x);

    uint32_t rows_ = 0;
    uint32_t cols_ = 0;
    uint32_t size_ = 0;

    std::shared_ptr<uint8_t> images_;
    std::shared_ptr<uint8_t> labels_;

    uint32_t classification_num_ = 10;
}; //! class MnistLoader
} //! namespace z

#endif  //! ALCHEMY_UTIL_MNISTLOADER_H