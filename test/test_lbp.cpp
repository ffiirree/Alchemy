#include <iostream>
#include <bitset>
#include "alchemy.h"

using namespace alchemy;

int main()
{
    auto image = imread("red.png");
    Matrix gray;

    cvtColor(image, gray, BGR2GRAY);
    imshow("original", image);

    Matrix lbp_image, rlbp_image, rulbp_image;

    LBP(gray, lbp_image, 1, 8, GRAY_SCALE_INVARIANCE);
    LBP(gray, rlbp_image, 1, 8, GRAY_SCALE_INVARIANCE | ROTATION_INVARIANCE);
    LBP(gray, rulbp_image, 1, 8, GRAY_SCALE_INVARIANCE | ROTATION_INVARIANCE | UNIFORM_PATTERN);

    imshow("lbp", lbp_image);
    imshow("rlbp", rlbp_image);
    imshow("rulbp", rulbp_image);

    waitKey(0);

    return 0;
}