#include <iostream>

#include"config_default.h"
#include "zmatrix.h"

using namespace std;

Matrix test()
{
	//Matrix mat1(3, 2);
	////mat1.eye();
	////cout << mat1;
	////mat1.ones(5, 4);
	////cout << mat1;
	////Matrix mat2(mat1);
	////Matrix mat3;
	////mat3 = mat1;
	//mat1.eye();
	//cout << mat1;
	//Matrix mat2;
	//mat2 = mat1.clone();

	//if (mat1 == mat2)
	//	cout << mat2;

	/*Matrix mat1(3, 2);
	Matrix mat2;
	mat2 = { 1,2,3 };
	cout << mat2;
	double a[2] = { 1,2 };
	mat2(a, sizeof(a)/sizeof(double));

	if (mat1 == mat2)
		cout << "==" << endl;
	else
		cout << "!=" << endl;*/

	Matrix mat1(1,3);
	mat1 = { 2,2,3};
	Matrix mat2(1, 3);
	mat2 = { 3,2,2 };
	cout << mat1.cross(mat2);

	Matrix m3(3, 3);
	m3 = { 2,3,5,1,6,3,7,2,8 };
	cout << m3;
	cout << m3.t();

	Matrix m4(2, 3);
	m4 = { 2,3,4, 2,5,4 };
	Matrix m5(3, 2);
	m5 = { 3,5, 4,6,2,6 };
	cout << m4*m5;

	Matrix m6(2, 2);
	m6 = { 3,2,52,4 };
	Matrix m7(2, 2);
	m7 = { 2,6,3,8 };
	cout << m6 - m7;

	//Matrix mat2(3,3);
	//mat2 = { 3,2,2, 3, 4, 5, 7, 2, 4};


	//cout << mat2;
	//cout << mat2.t();

	return mat1;
}

int main(int argc, char *argv[])
{
	Matrix mat = test();

	system("pause");
	return 0;
}