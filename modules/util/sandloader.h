#ifndef ALCHEMY_SANDLOADER_H
#define ALCHEMY_SANDLOADER_H

#include <glog/logging.h>
#include "loader.h"

namespace alchemy {

template <typename T>
class SandLoader : public Loader {
public:
    SandLoader(const string &img_path, const string &label_path) 
    {
        this->classification_num_ = 1;

        loadImages(img_path);
        loadLabels(label_path);

        this->image_bytes_ = this->image_size_ * sizeof(T);
        this->label_bytes_ = this->label_size_ * sizeof(T);
    }

protected:
    void loadImages(const string& path) override;
    void loadLabels(const string& path) override;


    template <typename ReadType>
    void readImages(std::fstream& file, ReadType*);
    void readImages(std::fstream& file, char *);

    template <typename ReadType>
    void readLabels(std::fstream& file, ReadType*);
    void readLabels(std::fstream& file, char *);
};

template <typename T>
void SandLoader<T>::loadImages(const string &path)
{
    LOG(INFO) << "images";
    assert(!path.empty());

    std::fstream image_file(path, std::ios::in | std::ios::binary);
    LOG_IF(FATAL, !image_file.is_open()) << "Can't open the file: " << path;


    // 1.     32 bit integer             number of images
    uint32_t size{0};
    image_file.read(reinterpret_cast<char *>(&size), sizeof(uint32_t));
    // image.size == label.size
    LOG_IF(FATAL, this->size_ != 0 && size != this->size_) << "size_ != 0 && size != size_!";
    this->size_ = size;

    // 2.     32 bit integer              number of rows
    image_file.read(reinterpret_cast<char *>(&this->rows_), 4);

    // 3.     32 bit integer              number of columns
    image_file.read(reinterpret_cast<char *>(&this->cols_), 4);

    // 4.
    this->chs_ = 1;

    this->image_size_ = this->cols_ * this->rows_ * this->chs_;

    readImages(image_file, static_cast<T*>(nullptr));

    image_file.close();
}

template <typename T>
void SandLoader<T>::loadLabels(const string &path)
{
    assert(!path.empty());

    std::fstream label_file(path, std::ios::in | std::ios::binary);
    LOG_IF(FATAL, !label_file.is_open()) << "Can't open the file: " << path;

    // 32 bit integer  10000            number of items
    uint32_t size{0};
    label_file.read(reinterpret_cast<char *>(&size), 4);

    // image.size == label.size
    LOG_IF(FATAL, this->size_ != 0 && size != this->size_) << "size_ != 0 && size != size_!";
    this->size_ = size;

    this->label_size_ = this->classification_num_;

    readLabels(label_file, static_cast<T*>(nullptr));

    label_file.close();
}

template <typename T>
template <typename ReadType>
void SandLoader<T>::readImages(std::fstream &file, ReadType *) 
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
void SandLoader<T>::readImages(std::fstream &file, char *)
{
    auto image_total = this->image_size_ * this->size_ * sizeof(uint8_t);
    this->images_.reset(new uint8_t[image_total]);

    file.read(reinterpret_cast<char *>(this->images_.get()), image_total);
}

template <typename T>
template <typename ReadType>
void SandLoader<T>::readLabels(std::fstream &file, ReadType *) 
{
    auto total = this->size_ * this->classification_num_;

    shared_ptr<uint8_t> labels;
    labels.reset(new uint8_t[total]);
    this->labels_.reset(new uint8_t[total * sizeof(ReadType)]);

//    auto labels_ptr = labels.get();
//    memset(labels_ptr, 0, total);
//    for(size_t i = 0; i < this->size_; ++i) {
//        uint8_t label = 0;
//        // 0008     unsigned byte   ??               label
//        file.read(reinterpret_cast<char *>(&label), 1);
//        labels_ptr[ i * this->classification_num_ + label] = 1;
//    }

    file.read(reinterpret_cast<char *>(labels.get()), total);

    // to
    auto dst_ptr = reinterpret_cast<ReadType *>(this->labels_.get());
    for(uint32_t i = 0; i < total; ++i) {
        dst_ptr[i] = labels.get()[i] / 10.0;
    }
}

template <typename T>
void SandLoader<T>::readLabels(std::fstream &file, char *) 
{
    LOG(FATAL) << "x";
//    auto total = this->size_ * this->classification_num_;
//
//    this->labels_.reset(new uint8_t[total]);
//    auto labels_ptr = this->labels_.get();
//    memset(labels_ptr, 0, total * sizeof(uint8_t));
//    for(size_t i = 0; i < this->size_; ++i) {
//        uint8_t label = 0;
//        // 0008     unsigned byte   ??               label
//        file.read(reinterpret_cast<char *>(&label), 1);
//        labels_ptr[ i * this->classification_num_ + label] = 1;
//    }
}
}

#endif //ALCHEMY_SANDLOADER_H
