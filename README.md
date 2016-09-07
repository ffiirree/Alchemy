# _Matrix

_Matrix 是一个使用C++编写的矩阵运算库，目的是为了辅助进行图像处理，因此里面会添加一些用于图像处理的矩阵运算函数。

<br><br>

# 版本
`V1.0`

<br><br>

# 特色
* 使用了类模板，可以较好的支持`double`、`float`、`signed int`、`short`、`char`以及`unsigned int`、`short`、`char`<br>
* 接近`Matlab`式的使用方式，上手简单<br>
* 使用引用计数，实现了自动内存管理，不需要为内存问题担心<br>
* 轻易和`openCV`中的`Mat`类型相互转化，更好的辅助学习`openCV`

<br><br>

# Usage
`_Matrix`是一个模板类，在`zmatrix.h`文件中，针对不同的数据类型重命名：
```
typedef _Matrix<double>             Matrix;
typedef _Matrix<double>             Matrix64f;
typedef _Matrix<float>              Matrix32f;
typedef _Matrix<signed int>         Matrix32s;
typedef _Matrix<unsigned int>       Matrix32u;
typedef _Matrix<signed short>       Matrix16s;
typedef _Matrix<unsigned short>     Matrix16u;
typedef _Matrix<signed char>        Matrix8s;
typedef _Matrix<unsigned char>      Matrix8u;
```

从上面可以看出`Matrix`是`double`类型的矩阵。

## 定义一个矩阵
```
// 定义一个2x2的矩阵，并赋值
Matrix m1(2, 2);
m1 = {2, 3,
     3, 6};

// 这只是对m1的浅复制
Matrix m2(m1);

// 定义1xn的矩阵
Matrix m3;
m3 = {4, 6, 6};
```

<br><br>

# 已经实现功能

## 矩阵的基本功能
* 转置，函数t()<br>
* 矩阵乘法，重载运算符*<br>
* 矩阵加法，重载运算符+<br>
* 矩阵减法，重载运算符-<br>
* 矩阵比较，重载运算符==和!=<br>
* 向量叉乘，即两个(1,3)矩阵的叉积，函数cross()<br>
* 卷积，函数conv()<br>
* 数乘，重载运算符+、-、*<br>
* 矩阵的迹，函数tr()<br>
* 矩阵点乘，函数dot()<br>

## 图像处理相关
* 和openCV中Mat的转换
* 卷积运算，可以实现简单的滤波和边缘检测功能

<br><br>

# 即将实现的功能

* 求逆<br>
* 求秩<br>


