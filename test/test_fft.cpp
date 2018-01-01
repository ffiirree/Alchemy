// Test dft fuction.
// Compute convolution.
#include "zmatrix.h"

int main()
{
    auto image = z::imread("test.jpeg");
    z::Matrix gray;
    z::cvtColor(image, gray, z::BGR2GRAY);
    z::Matrix64f OutputImage;

    z::dft(gray, OutputImage);

    z::Matrix64f ker(3, 3), kernelR, OutputKernel;
    ker = {
        0,  1, 0,
        1, -4, 1,
        0,  1, 0
    };
    z::copyMakeBorder(ker, kernelR, 0, image.rows - 3, 0, image.cols - 3);
    z::dft(kernelR, OutputKernel);

    z::Matrix64f BackImage(image.rows, image.cols, 2, z::Scalar{ 0, 0 });
    for (auto i = 0; i < image.rows; ++i) {
        for (auto j = 0; j < image.cols; ++j) {
            BackImage.at<z::Complex>(i, j) = OutputImage.at<z::Complex>(i, j) * OutputKernel.at<z::Complex>(i, j);
        }
    }

    z::Matrix64f DstImage;
    z::idft(BackImage, DstImage);

    z::Matrix res = DstImage;
    z::imshow("res", res);
    z::waitKey(0);

    return 0;
}