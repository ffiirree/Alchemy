#include <iostream>

#include"config_default.h"
#include "zmatrix.h"

using namespace std;

Matrix16s test()
{
	// 卷积测试
	Matrix16s m1(5, 5);
	m1 = { 
		5,7,9,1,5,
		3,6,7,1,98,
		4,2,96,4,3,
		5,4,9,56,3,
		5,1,5,7,61
	};

	cout << "m1.tr() =" << m1.tr() << endl;

	Matrix16s core(5, 5);
	core = {
		1,1,1,1,1,
		1,1,1,1,1,
		1,1,-1,1,1,
		1,1,1,1,1,
		1,1,1,1,1
	};

	cout << "core.tr() =" << core.tr() << endl;

	cout << m1.conv(core);

	return m1;
}

int main(int argc, char *argv[])
{
	Matrix16s mat = test();

	system("pause");
	return 0;
}