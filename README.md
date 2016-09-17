# _Matrix

_Matrix 是一个使用C++编写的矩阵运算库，目的是为了辅助进行图像处理，因此里面会添加一些用于图像处理的矩阵运算函数。

<br><br>

# 版本
`V 1.2.0 Beta`

<br><br>

# 特色
* 使用了类模板，可以较好的支持`double`、`float`、`signed int`、`short`、`char`以及`unsigned int`、`short`、`char`<br>
* 接近`Matlab`式的使用方式，上手简单<br>
* 使用引用计数，实现了自动内存管理，不需要为内存问题担心<br>
* 轻易和`openCV`中的`Mat`类型相互转化，更好的辅助学习`openCV`

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
### 基础
* 和openCV中Mat的转换
* 彩色图像转化为灰度图，`z::cvtColor(src, dst, BGR2GRAY);`

### 线性滤波
* 方框滤波函数`boxFilter()`
* 均值滤波函数`blur()`，彩色和灰度图像都可以
* 高斯滤波函数`GassionBlur()`

### 非线性滤波
* 中值滤波函数`medianBlur()`，彩色和灰度图像都可以

### 形态学滤波
* 腐蚀和膨胀函数`erode()/dilate()`
* 开运算、闭运算、顶帽、黑帽、形态学梯度运算，函数`morpEX()`

## 图像变换
### 边缘检测
* sobel算子
* 简单的canny边缘检测（sobel算子），彩色和灰度均可，效果不好

## Kinect 4 Windows v2 （Events）
* Kinect获取深度、彩色和红外图像，返回Mat类型的数据

<br><br>

# 即将实现的功能

* 求逆<br>
* 求秩<br>




# BUG日志

* 2016-09-07：将迹的返回类型同一改为double，防止结果溢出
* 2016-09-08：注意不要用Matrix的引用作参数返回值，这个问题需要好好解决
* blur(), 内核大小为（5， 5，runtime(openCV:27ms):4277ms -> 2964ms -> 728ms;<br>// 受不了openCV的运行效率了，太高了(这是在DEBUG模式下，在Release下要快的多，94ms，opencv 7ms)
* medianFilter(), 内核大小为（5， 5），runtime(openCV:334ms):21822ms - > 17594ms
* GassionBlur(), 内核大小为（5， 5），runtime(openCV:116ms):4245ms
* FFT卷积？？
* erode()/dilate(), 内核大小为（5， 5）,runtime(openCV:10ms):17555ms -> 920ms(release:140ms, 2ms)
* 2016-09-16：修改sobel局部分量的估计没有乘尺度因子
* 2016-09-16：修改zMatrix中conv上次更改后出现数据类型的错误
* 2016-09-17: 修改KinectSensor析构函数中释放指针时未检查是否为空的BUG