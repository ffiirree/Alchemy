#include "mnistloader.h"
#include <fstream>
#include <glog/logging.h>

namespace alchemy  {
MnistLoader::MnistLoader(const std::string &images_path, const std::string &labels_path)
{
    loadImages(images_path);
    loadLabels(labels_path);
}

uint32_t MnistLoader::reverse(uint32_t x)
{
    return (x >> 24) | ((x >> 8) & 0xff00) | ((x << 8) & 0xff0000) | ((x << 24) & 0xff);
}


void MnistLoader::loadImages(const std::string &path)
{
    assert(!path.empty());

    std::fstream image_file(path, std::ios::in | std::ios::binary);
    LOG_IF(FATAL, !image_file.is_open()) << "Can't open the file: " << path;

    // Now, read the data.
    // 0000     32 bit integer  0x00000803(2051) magic number
    uint32_t magic_number = 0;
    image_file.read(reinterpret_cast<char *>(&magic_number), 4);
    magic_number = reverse(magic_number);

    LOG_IF(FATAL, IMAGE_MAGIC_NUMBER != magic_number) << "IMAGE_MAGIC_NUMBER error!";

    // 0004     32 bit integer  60000            number of images
    uint32_t size;
    image_file.read(reinterpret_cast<char *>(&size), 4);
    size = reverse(size);
    // image.size == label.size
    LOG_IF(FATAL, size_ != 0 && size != size_) << "size_ != 0 && size != size_!";
    size_ = size;

    // 0008     32 bit integer  28               number of rows
    image_file.read(reinterpret_cast<char *>(&rows_), 4);
    rows_ = reverse(rows_);

    // 0012     32 bit integer  28               number of columns
    image_file.read(reinterpret_cast<char *>(&cols_), 4);
    cols_ = reverse(cols_);

    auto image_size = cols_ * rows_;
    auto image_total = image_size * size_;
    images_.reset(new uint8_t[image_total]);

    image_file.read(reinterpret_cast<char *>(images_.get()), image_total);

    image_file.close();
}


void MnistLoader::loadLabels(const std::string &path)
{
    assert(!path.empty());

    std::fstream label_file(path, std::ios::in | std::ios::binary);
    LOG_IF(FATAL, !label_file.is_open()) << "Can't open the file: " << path;

    // 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    uint32_t magic_number = 0;
    label_file.read(reinterpret_cast<char *>(&magic_number), 4);
    magic_number = reverse(magic_number);

    LOG_IF(FATAL, LABEL_MAGIC_NUMBER != magic_number) << "LABEL_MAGIC_NUMBER error!";

    // 32 bit integer  10000            number of items
    uint32_t size = 0;
    label_file.read(reinterpret_cast<char *>(&size), 4);
    size = reverse(size);
    // image.size == label.size
    LOG_IF(FATAL, size_ != 0 && size != size_) << "size_ != 0 && size != size_!";
    size_ = size;

    labels_.reset(new uint8_t[size_ * classification_num_]);
    auto labels_patr = labels_.get();
    memset(labels_patr, 0, size_ * classification_num_ * sizeof(uint8_t));
    for(size_t i = 0; i < size_; ++i) {
        uint8_t label = 0;
        // 0008     unsigned byte   ??               label
        label_file.read(reinterpret_cast<char *>(&label), 1);
        labels_patr[ i * classification_num_ + label] = 1;
    }
    label_file.close();
}

} //! namespace z

