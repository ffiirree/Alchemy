#include <alchemy.h>

int main()
{
    auto image = alchemy::imread("test.jpeg");
    alchemy::Matrix gray;
    alchemy::cvtColor(image, gray, alchemy::BGR2GRAY);
    alchemy::Matrix64f OutputImage;

    alchemy::dft(gray, OutputImage);

    alchemy::Matrix64f ker(3, 3), kernelR, OutputKernel;
    ker = {
        0,  1, 0,
        1, -4, 1,
        0,  1, 0
    };
    alchemy::copyMakeBorder(ker, kernelR, 0, image.rows - 3, 0, image.cols - 3);
    alchemy::dft(kernelR, OutputKernel);

    alchemy::Matrix64f BackImage(image.rows, image.cols, 2, alchemy::Scalar{ 0, 0 });
    for (auto i = 0; i < image.rows; ++i) {
        for (auto j = 0; j < image.cols; ++j) {
            BackImage.at<alchemy::Complex>(i, j) = OutputImage.at<alchemy::Complex>(i, j) * OutputKernel.at<alchemy::Complex>(i, j);
        }
    }

    alchemy::Matrix64f DstImage;
    alchemy::idft(BackImage, DstImage);

    alchemy::Matrix res = DstImage;
    alchemy::imshow("res", res);
    alchemy::waitKey(0);

    return 0;
}