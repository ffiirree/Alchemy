# Alchemy

Alchemy 是一个使用C++编写的CV库，包含传统算法和机器学习算法。
<br><br>

## 依赖
### 必须
- libjpeg: 读取jpeg图片， `sudo apt install libjpeg8-dev`
- libpng/zlib
- fftw: 进行快速傅里叶变换，`sudo apt install libfftw3-dev`
- gtk-2.x: 显示图片使用， `sudo apt install libgtk2.0-dev`
- FFmpeg: 读取摄像头/视频数据， `sudo apt install ffmpeg`
- BLAS: ATLAS, `sudo apt install libatlas-base-dev`
- CUDA 9.0/cuDNN 7: GPU计算，安装见官网
- Boost: C++通用库，`sudo apt install libboost-all-dev`
- Glog: 日志，方便调试， `sudo apt install libgoogle-glog-dev`
- [NNPACK](https://github.com/Maratyszcza/NNPACK), CPU快速计算卷积，安装见github
- [MathGL2](http://mathgl.sourceforge.net/doc_en/Main.html): 数据可视化，使用`CMake`从源码编译安装
### 可选
- OpenCV

<br>

## CMake构建
```
mkdir build
cd build
cmake ..
```

<br>

## Todo
 - [x] 保存图像的基本类:`_Matrix`
 - [ ] 使用`BLAS`重写矩阵运算
 - [ ] 分别实现`CPU/GPU`版本的矩阵运算
 - [x] jpeg图像**读写**
 - [x] png图像数据**读取**
 - [x] 摄像头(Linux:v4l2)数据读取
 - [x] mp4等视频数据读取
 －[x] `VideoCapture` 读取Gif图片
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
 - [x] 256 LBP/ELBP　(TODO:　任意r和P)
 - [ ] Haar
 - [ ] HOG
 - [ ] SIFT
### 
 - [x] 实现张量类`Tensor`(也就是N维数组)
 - [x] 调整结构为: Tensor -> Layer -> Network -> Optimizer 结构
 - [x] weight/bias初始化方法: normal/uniform/xavier
 - [ ] GPU版本实现(已部分实现)
 - [x] L1/L2正则化
 - [x] Accuracy Layer
 - [x] Convolution Layer
 - [x] Pooling Layer
 - [x] Inner Product
 - [x] Dropout Layer
 - [x] ReLU Layer
 - [x] sigmoid Layer
 - [x] tanh Layer
 - [ ] Softmax Layer
 - [x] Softmax Loss Layer
 - [x] Euclidean Loss Layer
 - [x] Sigmoid cross-entropy Loss Layer
 - [x] SGD Optimizer
 - [ ] AdaDelta Optimizer
 - [ ] Nesterov Optimizer
 - [ ] Adam Optimizer