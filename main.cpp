#include <iostream>
#include <opencv2\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <string>
#include <ctime>  

#include "zcore\zcore.h"
#include "zimgproc\zimgproc.h"
#include "zimgproc\transform.h"
#include "zgui\zgui.h"
#include "zcore\debug.h"

using namespace std;
using namespace z;

int main(int argc, char *argv[])
{
	z::Matrix test(4, 8);
	test = { 0, 1, 2, 3, 4, 5, 6, 7,
		1, 2, 3, 4, 5, 6, 7, 0,
		2, 3, 4, 5, 6, 7, 0, 1,
		3, 4, 5, 6, 7, 0, 1, 2 };

	// 输出测试数据
	cout << "Test Matrix is:" << endl << test << endl;

	z::Matrix dft_test = test.clone();
	z::Matrix dft_dst, idft_dst;

	// 普通离散傅里叶变换
	z::dft(dft_test, dft_dst);
	cout << "z_dft = " << endl << dft_dst << endl;

	z::idft(dft_dst, idft_dst);
	cout << "z_idft = " << endl << idft_dst << endl;

	// 快速傅里叶变换
	// 结果应该和DFT的结果一样(会有很小的差别)
	z::Matrix dst;
	z::fft(test, dst);
	cout << "z_fft = " << endl << dst << endl;

	getchar();

	return 0;
}