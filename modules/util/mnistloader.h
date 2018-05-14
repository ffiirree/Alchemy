#ifndef ALCHEMY_UTIL_MNISTLOADER_H
#define ALCHEMY_UTIL_MNISTLOADER_H

#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <glog/logging.h>
#include "core/matrix.h"
#include "loader.h"

namespace alchemy {

template <typename T>
class MnistLoader : public Loader {
public:
    enum {
        LABEL_MAGIC_NUMBER = 2049, IMAGE_MAGIC_NUMBER = 2051
    };

public:
    MnistLoader(const std::string &images_path, const std::string &labels_path);

private:
    void loadImages(const std::string &path) override;
    void loadLabels(const std::string &path) override;

    template <typename ReadType>
    void readImages(std::fstream& file, ReadType*);
    void readImages(std::fstream& file, char *);

    template <typename ReadType>
    void readLabels(std::fstream& file, ReadType*);
    void readLabels(std::fstream& file, char *);

    static uint32_t reverse(uint32_t x);
}; //! class MnistLoader

template <typename T>
MnistLoader<T>::MnistLoader(const std::string &images_path, const std::string &labels_path)
{
    this->classification_num_ = 10;

    loadImages(images_path);
    loadLabels(labels_path);

    this->image_bytes_ = this->image_size_ * sizeof(T);
    this->label_bytes_ = this->label_size_ * sizeof(T);
}

template <typename T>
uint32_t MnistLoader<T>::reverse(uint32_t x)
{
    return (x >> 24) | ((x >> 8) & 0xff00) | ((x << 8) & 0xff0000) | ((x << 24) & 0xff);
}

template <typename T>
void MnistLoader<T>::loadImages(const std::string &path)
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
    LOG_IF(FATAL, this->size_ != 0 && size != this->size_) << "size_ != 0 && size != size_!";
    this->size_ = size;

    // 0008     32 bit integer  28               number of rows
    image_file.read(reinterpret_cast<char *>(&this->rows_), 4);
    this->rows_ = reverse(this->rows_);

    // 0012     32 bit integer  28               number of columns
    image_file.read(reinterpret_cast<char *>(&this->cols_), 4);
    this->cols_ = reverse(this->cols_);

    // chs
    this->chs_ = 1;

    // size
    this->image_size_ = this->cols_ * this->rows_ * this->chs_;

    readImages(image_file, static_cast<T*>(nullptr));

    image_file.close();
}

template <typename T>
template <typename ReadType>
void MnistLoader<T>::readImages(std::fstream &file, ReadType *)
{
    auto image_total = this->image_size_ * this->size_;
    shared_ptr<uint8_t> images;
    images.reset(new uint8_t[image_total]);
    this->images_.reset(new uint8_t[image_total * sizeof(ReadType)]);

    file.read(reinterpret_cast<char *>(images.get()), image_total);

    // to
    auto image_ptr = reinterpret_cast<ReadType*>(this->images_.get());
    for(uint32_t i = 0; i < image_total; ++i) {
        image_ptr[i] = images.get()[i];
    }
}

template <typename T>
void MnistLoader<T>::readImages(std::fstream &file, char *)
{
    auto image_total = this->image_size_ * this->size_ * sizeof(char);
    this->images_.reset(new uint8_t[image_total]);

    file.read(reinterpret_cast<char *>(this->images_.get()), image_total);
}

template <typename T>
void MnistLoader<T>::loadLabels(const std::string &path)
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
    LOG_IF(FATAL, this->size_ != 0 && size != this->size_) << "size_ != 0 && size != size_!";
    this->size_ = size;

    this->label_size_ = this->classification_num_;

    readLabels(label_file, static_cast<T*>(nullptr));

    label_file.close();
}

template <typename T>
template <typename ReadType>
void MnistLoader<T>::readLabels(std::fstream &file, ReadType *)
{
    auto total = this->size_ * this->classification_num_;

    shared_ptr<uint8_t> labels;
    labels.reset(new uint8_t[total]);
    this->labels_.reset(new uint8_t[total * sizeof(ReadType)]);

    auto labels_ptr = labels.get();
    memset(labels_ptr, 0, total);
    for(size_t i = 0; i < this->size_; ++i) {
        uint8_t label = 0;
        // 0008     unsigned byte   ??               label
        file.read(reinterpret_cast<char *>(&label), 1);
        labels_ptr[ i * this->classification_num_ + label] = 1;
    }

    // to
    auto dst_ptr = reinterpret_cast<ReadType *>(this->labels_.get());
    for(uint32_t i = 0; i < total; ++i) {
        dst_ptr[i] = labels_ptr[i];
    }
}

template <typename T>
void MnistLoader<T>::readLabels(std::fstream &file, char *)
{
    auto total = this->size_ * this->label_size_ * sizeof(char);

    this->labels_.reset(new uint8_t[total]);
    auto labels_ptr = this->labels_.get();
    memset(labels_ptr, 0, total);
    for(size_t i = 0; i < this->size_; ++i) {
        uint8_t label = 0;
        // 0008     unsigned byte   ??               label
        file.read(reinterpret_cast<char *>(&label), 1);
        labels_ptr[ i * this->classification_num_ + label] = 1;
    }
}
} //! namespace alchemy

#endif  //! ALCHEMY_UTIL_MNISTLOADER_H