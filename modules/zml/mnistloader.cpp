#include "mnistloader.h"
namespace z
{

MnistLoader::MnistLoader(const std::string &images_path, const std::string &labels_path) {
    if(!loadImages(images_path) || !loadLabels(labels_path))
        size_ = 0;
}

uint32_t MnistLoader::reverse(uint32_t x) {
    return (x >> 24) | ((x >> 8) & 0xff00) | ((x << 8) & 0xff0000) | ((x << 24) & 0xff);
}


bool MnistLoader::loadImages(const std::string &path) {
    assert(!path.empty());

    std::fstream image_file(path, std::ios::in | std::ios::binary);
    if(!image_file.is_open()) return false;

    // Now, read the data.
    // 0000     32 bit integer  0x00000803(2051) magic number
    uint32_t magic_number = 0;
    image_file.read(reinterpret_cast<char *>(&magic_number), 4);
    magic_number = reverse(magic_number);

    if(IMAGE_MAGIC_NUMBER != magic_number) return false;

    // 0004     32 bit integer  60000            number of images
    uint32_t size;
    image_file.read(reinterpret_cast<char *>(&size), 4);
    size = reverse(size);
    // image.size == label.size
    if(size_ != 0 && size != size_) return false;
    size_ = size;

    // 0008     32 bit integer  28               number of rows
    image_file.read(reinterpret_cast<char *>(&rows_), 4);
    rows_ = reverse(rows_);

    // 0012     32 bit integer  28               number of columns
    image_file.read(reinterpret_cast<char *>(&cols_), 4);
    cols_ = reverse(cols_);

    // read images
    for(uint32_t i = 0; i < size_; ++i) {
        Pair pair;
        pair.first.create(rows_, cols_, 1);
        image_file.read(reinterpret_cast<char *>(pair.first.data), pair.first.total());
        data_.push_back(pair);
    }

    image_file.close();
    return true;
}


bool MnistLoader::loadLabels(const std::string &path) {
    assert(!path.empty());

    std::fstream label_file(path, std::ios::in | std::ios::binary);
    if(!label_file.is_open()) return false;

    // 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    uint32_t magic_number = 0;
    label_file.read(reinterpret_cast<char *>(&magic_number), 4);
    magic_number = reverse(magic_number);

    if(LABEL_MAGIC_NUMBER != magic_number) return false;

    // 32 bit integer  10000            number of items
    uint32_t size = 0;
    label_file.read(reinterpret_cast<char *>(&size), 4);
    size = reverse(size);
    // image.size == label.size
    if(size_ != 0 && size != size_) return false;
    size_ = size;

    for(size_t i = 0; i < size_; ++i) {
        // 0008     unsigned byte   ??               label
        label_file.read(reinterpret_cast<char *>(&(data_.at(i).second)), 1);
    }
    label_file.close();

    return true;
}

} //! namespace z

