#include "zmatrix.h"


int main(int argc, char * argv[])
{
    auto image = z::imread(const_cast<char *>("test.jpeg"));
    z::Matrix gray = image;
    z::cvtColor(image, gray, z::BGR2GRAY);


    z::imshow("hello", gray);
    z::imshow("original", image);
    z::waitKey(0);

    return 0;
}