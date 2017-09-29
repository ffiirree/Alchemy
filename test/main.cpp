#include "zmatrix.h"


int main()
{
    auto image = z::imread("test.jpeg");

    TimeStamp timer;
    timer.start();

    z::Matrix bimage = image > 75;
    image += image;
    std::cout << timer.runtime() << std::endl;
    
    getchar();
    return 0;
}