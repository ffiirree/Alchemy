#include <iostream>
#include <opencv2\opencv.hpp>

#include "zcore\zcore.h"
#include "zimgproc\zimgproc.h"
#include "zimgproc\transform.h"
#include "zgui\zgui.h"
#include "zcore\debug.h"

int main(int argc, char *argv[])
{
    z::Matrix test(4, 8);
    test = { 
        0, 1, 2, 3, 4, 5, 6, 7,
        1, 2, 3, 4, 5, 6, 7, 0,
        2, 3, 4, 5, 6, 7, 0, 1,
        3, 4, 5, 6, 7, 0, 1, 2 
    };

    // 输出测试数据
    std::cout << "Test Matrix is:" << std::endl << test << std::endl;

    z::Matrix dft_test = test.clone();
    z::Matrix dft_dst, idft_dst;

    // 普通离散傅里叶变换
    z::dft(dft_test, dft_dst);
    std::cout << "z_dft = " << std::endl << dft_dst << std::endl;

    z::idft(dft_dst, idft_dst);
    std::cout << "z_idft = " << std::endl << idft_dst << std::endl;

    // 快速傅里叶变换
    // 结果应该和DFT的结果一样(会有很小的差别)
    z::Matrix fft_dst;
    z::fft(test, fft_dst);
    std::cout << "z_fft = " << std::endl << fft_dst << std::endl;

    z::Matrix ifft_dst;
    z::ifft(fft_dst, ifft_dst);
    std::cout << "z_ifft = " << std::endl << ifft_dst << std::endl;

    system("pause");
	return 0;
}