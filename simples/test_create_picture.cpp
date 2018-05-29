#include "alchemy.h"
#include <fstream>

using namespace std;
using namespace alchemy;

int main()
{
    fstream image_file, label_file;
    image_file.open("/home/ffiirree/Code/Alchemy/resources/test_images_10k.ubyte", ios::out|ios::binary);
    label_file.open("/home/ffiirree/Code/Alchemy/resources/test_labels_10k.ubyte", ios::out|ios::binary);
    if(!image_file.is_open() || !label_file.is_open()) {
        cout << "Open file failure.";
        return -1;
    }

    const int CLASS = 10;
    const int NPC = 1000;
    const uint32_t NUM = CLASS * NPC;
    const uint32_t COLS = 100;
    const uint32_t ROWS = 100;

    image_file.write(reinterpret_cast<const char *>(&NUM), sizeof(uint32_t));
    image_file.write(reinterpret_cast<const char *>(&ROWS), sizeof(uint32_t));
    image_file.write(reinterpret_cast<const char *>(&COLS), sizeof(uint32_t));

    label_file.write(reinterpret_cast<const char *>(&NUM), sizeof(uint32_t));

    Matrix matrix({25, 25, 1}, Scalar{0});

    for(auto n = 0; n < NPC; ++n) {
        for(uint8_t p = 0; p < CLASS; ++p) {

            double probability = p * 0.1;

            // one picture
            Filler<CPU, uint8_t>::bernoulli_fill(static_cast<int>(matrix.count()), matrix.ptr(), probability);
            matrix *= 255;

            Matrix image({ROWS, COLS, 1}, Scalar{0});

            for(auto row = 0; row < matrix.rows_; ++row) {
                for(auto col = 0; col < matrix.cols_; ++col) {

                    for(auto i = 0; i < 4; ++i) {
                        for(auto j = 0; j < 4; ++j) {
                            image.at(4 * row + i, 4 * col + j) = matrix.at(row, col);
                        }
                    }
                }
            }

            Matrix result;
            GaussianBlur(image, result, {5, 5});

            image_file.write(reinterpret_cast<const char *>(result.ptr_), sizeof(char) * result.count());
            label_file.write(reinterpret_cast<const char *>(&p), sizeof(uint8_t));
        }
    }


    image_file.close();
    label_file.close();

    return 0;
}