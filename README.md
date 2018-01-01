# zMatrix

zMatrix 是一个使用C++编写的计算机视觉库，并且添加一些机器学习算法。
<br><br>

## 特色
* **C++14**
* 尽可能的提高**可读性**
* zMatrix使用**类模板**来支持各种类型
* 函数接口尽量和`opencv`类似，zMatrix类可以和openCV中的Mat类转换
* `z::Matrix`类使用**引用计数**来达到**自动内存管理**的目的，不需要为内存问题担心
* 实现较为简单，可以用来学习和实现一些简单的视觉算法，加深对算法的理解
* 辅助学习`OpenCV`

<br>

## 依赖
### 必须
- libjpeg
- fftw
- gtk-2.x
- FFmpeg(libavdevice libavutil libavcodec libavformat libswscale)
- BLAS: ATLAS
- CUDA 9.0/cuDNN 7
- Boost
- Glog
### 可选
- Kinect
- OpenCV

<br>

## 使用CMake生成工程
```
mkdir build
cd build
cmake ..
```
<br>

## 测试环境
### Linux
```
Ubuntu 16.04
CLion 2017.3
GCC&G++ 5.4
CMake 3.9.6
```
Ubuntu的一些开发软件配置安装，可以参照一下我博客[Ubuntu 16.04 部分专业软件安装](http://blog.csdn.net/ice__snow/article/details/53958765)

<br><br>

## Todo
 - [x] 保存图像的基本类:`_Matrix`
 - [ ] 使用`BLAS`重写矩阵运算
 - [ ] 分别实现`CPU/GPU`版本的矩阵运算
 - [x] jpeg图像读写
 - [ ] png等图像数据读取
 - [x] 摄像头(Linux:v4l2)数据读取
 - [ ] 视频数据读取
 - [ ] 再次支持`Windows`(主要VS好用)
 - [x] ROI
### 
 - [x] 颜色空间转换:`cvtColor()`
 - [x] 线性滤波:`boxFilter()/blur()/GassionBlur()`
 - [x] 非线性滤波: `medianBlur()/bilateralFilter()`
 - [x] 形态学滤波: `morpEX()`
 - [x] 图像轮廓: `findContours()/findOutermostContours()`
 - [x] 阈值: `threshold()`
 - [x] DFT: `FFTW`实现(自己实现的后来更新_Matrix类后出现问题了，没有修改)
 - [x] 图像金字塔: `pyrUp()/pyrDown()`
 - [x] `canny/sobel`边缘检测(实现简单，效果不好)
 - [ ] 仿射变换
 - [ ] SIFT
### 
 - [x] 3层神经网络识别手写数字
 - [x] 实现张量类`Tensor`(也就是N维数组)
 - [x] 调整结构为: Tensor -> Layer -> Network -> Optimization 结构
 - [ ] Convolution Layer
 - [ ] Pooling Layer
 - [x] Inner Product
 - [ ] Dropout Layer
 - [ ] ReLU Layer
 - [x] sigmoid Layer
 - [ ] tanh Layer
 - [ ] Softmax Layer
 - [x] Euclidean Loss Layer
 - [ ] Cross-entropy Loss Layer
 - [x] SGD
 - [ ] AdaDelta
 - [ ] Nesterov
 - [ ] Adam

<br><br>

## Visual Studio插件推荐
`Image Watch`:可以在新窗口以图片的形式显示`cv::Mat`类型数据，非常方便调试。VS2017需要下载修改过的版本，官方版本支持到2015。<br>
如果要支持`z::Matrix`类型的显示，使用`resources/ImageWatch/ImageWatchOpenCV.natvis`替换插件的配置文件。<br>
如果要显示自定义数据结构，Visual studio官网查看natvis文件的配置格式。

`ForceUTF8(no BOM)`:文件保存为utf8格式
