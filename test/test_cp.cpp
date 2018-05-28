#include <alchemy.h>
#include <algorithm>

using namespace alchemy;

int main()
{
    Matrix image(100, 100, 1);
    Filler<uint8_t>::bernoulli_fill(static_cast<int>(image.size()), image.ptr_, 0.01);

    std::for_each(std::begin(image), std::end(image), [](auto&& value){
        if(value == 1) value = 255;
    });

    Matrix image1(100, 100, 1);
    Filler<uint8_t>::bernoulli_fill(static_cast<int>(image1.size()), image1.ptr_, 0.01);

    std::for_each(std::begin(image1), std::end(image1), [](auto&& value){
        if(value == 1) value = 255;
    });


    imshow("image", image);
    imshow("image1", image1);
    waitKey(0);

    return 0;
}