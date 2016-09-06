#include <iostream>

#include"config_default.h"
#include "zmatrix.h"

using namespace std;

Matrix test()
{
	// ¾í»ý²âÊÔ
	Matrix m1(5, 5);
	m1 = { 
		5,7,9,1,5,
		3,6,7,1,98,
		4,2,96,4,3,
		5,4,9,56,3,
		5,1,5,7,61
	};
	cout << "m1 = " << endl << m1;

	Matrix core(3, 3);
	core = {
		1,1,1,
		1,-1,1,
		1,1,1
	};
	cout << "core = " << endl << core;


	cout << "conv = " << endl << m1.conv(core);

	return m1;
}

int main(int argc, char *argv[])
{
	Matrix mat = test();

	system("pause");
	return 0;
}