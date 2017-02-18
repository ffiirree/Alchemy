#include <iostream>
#include <opencv2\opencv.hpp>
#include <fstream>

#include "zcore\zcore.h"
#include "zimgproc\zimgproc.h"
#include "zimgproc\transform.h"
#include "zgui\zgui.h"
#include "zcore\debug.h"

int main(int argc, char *argv[])
{
    z::Matrix8u test = z::imread("test.jpeg");
    z::Matrix8u gray;
    z::Matrix8u res_image;
    TimeStamp timer;
    

    //res_image = gray.clone();
    z::Matrix ker(3, 3);
    ker = {
        -1, -1, -1,
        -1,  9, -1,
        -1, -1, -1
    };
    z::cvtColor(test, gray, BGR2GRAY);
    z::Matrix8u good = gray;

    //gray.conv(ker, res_image);
    cv::imshow("original", cv::Mat(gray));
    z::Matrix fft_src, fft_dst, ifft_dst;
    z::Matrix8u  res;

    // ---------------------------------------------------------------------------------------------------------------------------
    // 使用fft进行卷积运算
    // fft
    fft_src = gray;

    timer.start();
    z::fft(fft_src, fft_dst);

    // 乘积
    //k_fft
    z::Matrix ker_;
    copyMakeBorder(ker, ker_, 0, fft_dst.rows - 3, 0, fft_dst.cols - 3);
    z::Matrix ker_i(ker_.rows, ker_.cols, 1);
    z::Matrix kernel;
    z::merge(ker_, ker_i, kernel);
    z::Matrix kernel_fft;
    z::fft(kernel, kernel_fft);

    z::Matrix ifft_src(kernel_fft.rows, kernel_fft.cols, 2);
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
    gray.conv(ker, res_good);
    std::cout << timer.runtime() << std::endl;
    cv::imshow("res_good", cv::Mat(res_good));
    // ---------------------------------------------------------------------------------------------------------------------------

    cv::waitKey(0);

    return 0;
}