# zMatrix

zMatrix 是一个使用C++编写的计算机视觉库，包含传统算法和机器学习算法。
<br><br>

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
- NNPACK
### 可选
- Kinect
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
 - [x] 实现张量类`Tensor`(也就是N维数组)
 - [x] 调整结构为: Tensor -> Layer -> Network -> Optimization 结构
 - [x] weight/bias初始化方法: normal/uniform/xavier
 - [ ] 正则化
 - [x] Accuracy Layer
 - [ ] Convolution Layer
 - [ ] Pooling Layer
 - [x] Inner Product
 - [ ] Dropout Layer
 - [ ] ReLU Layer
 - [x] sigmoid Layer
 - [x] tanh Layer
 - [ ] Softmax Layer
 - [x] Euclidean Loss Layer
 - [x] Sigmoid cross-entropy Loss Layer
 - [x] SGD
 - [ ] AdaDelta
 - [ ] Nesterov
 - [ ] Adam