#include <iostream>
#include <opencv2\opencv.hpp>
#include <fstream>
#include <cfloat>

#include "zmatrix.h"

// 边界处理还有问题
int main(int argc, char *argv[])
{
    z::Matrix8u test = z::imread("test.jpeg");
    z::Matrix8u gray;
    z::Matrix8u res_image;
    TimeStamp timer;

    z::Matrix64f ker(3, 3);
    ker = {
        0, 1, 0, 1, -4, 1, 0, 1, 0
    };
    z::cvtColor(test, gray, BGR2GRAY);
    z::Matrix8u good = gray;

    cv::imshow("original", cv::Mat(gray));
    z::Matrix64f fft_src, fft_dst, ifft_dst;
    z::Matrix8u  res;

    // ---------------------------------------------------------------------------------------------------------------------------
    // 使用fft进行卷积运算
    // fft
    fft_src = gray;

    timer.start();
    z::fft(fft_src, fft_dst);

    // 乘积
    //k_fft
    z::Matrix64f ker_, ker_ifft;
    copyMakeBorder(ker, ker_, 0, fft_dst.rows - 3, 0, fft_dst.cols - 3);
    z::Matrix64f kernel_fft;
    z::fft(ker_, kernel_fft);
    

    z::Matrix64f ifft_src(kernel_fft.rows, kernel_fft.cols, 2);
    for (int i = 0; i < fft_dst.rows; ++i) {
        for (int j = 0; j < fft_dst.cols; ++j) {
            ifft_src.ptr(i, j)[0] = fft_dst.ptr(i, j)[0] * kernel_fft.ptr(i, j)[0] - fft_dst.ptr(i, j)[1] * kernel_fft.ptr(i, j)[1];
            ifft_src.ptr(i, j)[1] = fft_dst.ptr(i, j)[1] * kernel_fft.ptr(i, j)[0] + fft_dst.ptr(i, j)[0] * kernel_fft.ptr(i, j)[1];
        }
    }

    // ifft
    z::ifft(ifft_src, ifft_dst);
    std::cout << timer.runtime() << std::endl;

    res = ifft_dst;
    std::vector<z::Matrix8u> mv;
    z::spilt(res, mv);
    z::Matrix8u res__(gray.rows, gray.cols, 1);
    for (int i = 0; i < gray.rows; ++i) {
        for (int j = 0; j < gray.cols; ++j) {
            res__.ptr(i, j)[0] = mv.at(0).ptr(i, j)[0];
        }
    }
    cv::imshow("res", cv::Mat(res__));
    // ---------------------------------------------------------------------------------------------------------------------------


    // ---------------------------------------------------------------------------------------------------------------------------
    // 普通卷积运算
    z::Matrix8u res_good;
    timer.start();
    res_good = gray.conv(ker);
    std::cout << timer.runtime() << std::endl;
    cv::imshow("res_good", cv::Mat(res_good));
    // ---------------------------------------------------------------------------------------------------------------------------

    cv::waitKey(0);

    return 0;
}