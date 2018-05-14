#ifndef ALCHEMY_LOADER_H
#define ALCHEMY_LOADER_H

#include <string>
#include <cassert>
#include "core/common.h"

namespace alchemy {

class Loader {
public:
    using DataPair = pair<void *, void *>;

public:
    Loader() = default;
    Loader(const Loader&) = delete;
    Loader&operator=(const Loader&) = delete;
    virtual ~Loader() = default;

    bool hasNext() { return idx_ < size_; }
    bool hasNext(int num) { return idx_ + num <= size_; }
    void reset() { idx_ = 0; }

    DataPair next()
    {
        assert(idx_ < size_);

        auto data_pair = std::make_pair(images_.get() + idx_ * image_bytes_, labels_.get() + idx_ * label_bytes_);
        idx_++;
        return data_pair;
    }

    DataPair next(int num)
    {
        assert(idx_ + num <= size_);

        auto data_pair = std::make_pair(images_.get() + idx_ * image_bytes_, labels_.get() + idx_ * label_bytes_);
        idx_ += num;
        return data_pair;
    }

    inline bool empty() const { return !size_; }

    inline auto size() const { return size_; }

    inline auto rows() const { return rows_; }
    inline auto cols() const { return cols_; }
    inline auto chs() const { return chs_; }
    inline auto image_size() const { return image_size_; }
    inline auto label_size() const { return label_size_; }

    inline auto classification_num() const { return classification_num_; }

    inline auto images() const { return images_; }
    inline auto labels() const { return labels_; }

protected:
    virtual void loadImages(const string &path) = 0;
    virtual void loadLabels(const string &path) = 0;

    shared_ptr<uint8_t> images_;
    shared_ptr<uint8_t> labels_;

    uint32_t rows_{0};
    uint32_t cols_{0};
    uint32_t chs_{1};
    uint32_t image_size_{0};
    uint32_t label_size_{0};

    uint32_t image_bytes_{0};
    uint32_t label_bytes_{0};

    uint32_t size_{0};

    uint32_t idx_{0};

    uint32_t classification_num_{0};
}; //! Loader

} // namespace alchemy

#endif //ALCHEMY_LOADER_H
