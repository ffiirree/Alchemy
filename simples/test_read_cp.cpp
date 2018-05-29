#include <opencv2/opencv.hpp>
#include <fstream>
#include "util/sandloader.h"

using namespace std;
using namespace cv;

int main()
{
    const string images_path = "/home/ffiirree/Code/Alchemy/resources/train_images_20k.ubyte";
    const string labels_path = "/home/ffiirree/Code/Alchemy/resources/train_labels_20k.ubyte";
//    alchemy::SandLoader<uint8_t> a(images_path, labels_path);
    fstream image_file, label_file;
    image_file.open(images_path, ios::in|ios::binary);
    label_file.open(labels_path, ios::in|ios::binary);
    if(!image_file.is_open() || !label_file.is_open()) {
        cout << "Open file failure.";
        return -1;
    }

    uint32_t number{0}, cols{0}, rows{0};
    image_file.read(reinterpret_cast<char *>(&number), sizeof(uint32_t));
    image_file.read(reinterpret_cast<char *>(&cols), sizeof(uint32_t));
    image_file.read(reinterpret_cast<char *>(&rows), sizeof(uint32_t));

    for(auto i = 0; i < 100; ++i) {
        Mat matrix(100, 100, CV_8UC1);
        image_file.read(reinterpret_cast<char *>(matrix.data), matrix.total() * sizeof(uint8_t));

        std::string name = "/home/ffiirree/Code/Alchemy/resources/images/" + std::to_string(i) + ".png";
        imwrite(name, matrix);
//        double label = 0;
//        label_file.read(reinterpret_cast<char *>(&label), sizeof(double));
    }

    image_file.close();
    label_file.close();

    return 0;
}