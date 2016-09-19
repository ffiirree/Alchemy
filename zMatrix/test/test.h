#ifndef _TEST_H
#define _TEST_H

#include"black.h"

#define ASSERT_FALSE(condition) 
#define ASSERT_TRUE(condition)

#define ASSERT_EQ(expected,actual)
#define ASSERT_NE(expected,actual)


bool test_all();

/*-------------------zMatrix类的测试-------------------*/
bool test_zmatrix();

/*-------------------jpeg文件读写-------------------*/
bool test_imread();
bool test_imwrite();

/*-------------------线性滤波函数-------------------*/
bool test_Gassion();
bool test_blur();
bool test_boxFilter();
bool test_GaussianBlur();
/*-------------------非线性滤波函数-------------------*/
bool test_medianFilter();

/*-------------------形态学滤波函数-------------------*/
bool test_morphOp();
bool test_erode();
bool test_dilate();

bool test_morphEx();
bool test_open();

/*-------------------图像变换-------------------*/

#endif // !_TEST_H