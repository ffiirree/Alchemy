/**
 ******************************************************************************
 * @file    transform.cpp
 * @author  zlq
 * @version V1.0
 * @date    2016.9.14
 * @brief   图像变换的函数实现
 ******************************************************************************
 * @attention
 *
 *
 ******************************************************************************
 */
#include <cmath>
#include "transform.h"
#include "zcore\debug.h"

namespace z {

/**
 * @declaration 函数声明
 *              内部函数
 */ 
static inline void only_max(Matrix8u&src, Matrix8u&dst, Matrix8u&srcGD);
static void double_threashold(Matrix8u&src, Matrix8u&dst, double threshold1, double threshold2);
static void sobel(Matrix8u&src, Matrix8u&dst, Matrix8u&dstGD, int dx = 1, int dy = 1, int ksize = 3, bool noGD = false);

/**
 * @berif Canny中的非极大值抑制
 */
inline void only_max(Matrix8u&src, Matrix8u&dst, Matrix8u&srcGD)
{
	unsigned char * srcptr, *dstptr;

	if (!src.equalSize(srcGD))
		throw std::runtime_error("src.equalSize(srcGD)!");

	dst = src.clone();
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			
			srcptr = src.ptr(i, j);
			dstptr = dst.ptr(i, j);

			for (int k = 0; k < src.chs; ++k) {

                switch (srcGD.ptr(i, j)[k])
                {
                case 0:  // [j - 1 | - | j + 1 ]
                    if (j - 1 >= 0 && j + 1 < src.cols
                        && srcptr[k] < src.ptr(i, j - 1)[k]
                        && srcptr[k] < src.ptr(i, j + 1)[k])
                        dstptr[k] = 0;
                    break;

                case 45: 
                    if ((i - 1 >= 0 && j - 1 >= 0 && i + 1 < src.rows && j + 1 < src.cols) 
                        && srcptr[k] < src.ptr(i - 1, j + 1)[k] 
                        && srcptr[k] < src.ptr(i + 1, j - 1)[k])
                        dstptr[k] = 0;
                    break;

                case 90:
                    if ((i - 1 >= 0 && i + 1 < src.rows)
                        && srcptr[k] < src.ptr(i - 1, j)[k]
                        && srcptr[k] < src.ptr(i + 1, j)[k])
                        dstptr[k] = 0;
                    break;

                case 135:
                    if ((i - 1 >= 0 && j - 1 >= 0 && i + 1 < src.rows && j + 1 < src.cols)
                        && srcptr[k] < src.ptr(i - 1, j - 1)[k]
                        && srcptr[k] < src.ptr(i + 1, j + 1)[k])
                        dstptr[k] = 0;
                    break;
                }
			} // for(k)
		} // for(j)
	} //  for(i)
}


/**
 * @berif Canny中的双阈值
 */
void double_threashold(Matrix8u&src, Matrix8u&dst, double threshold1, double threshold2)
{
	double maxt = threshold1 > threshold2 ? threshold1 : threshold2;
	double mint = threshold1 < threshold2 ? threshold1 : threshold2;

	if (!dst.equalSize(src))
		dst.create(src.rows, src.cols, src.chs);

	unsigned char *ptr, * dstPtr;

	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			ptr = src.ptr(i, j);
			dstPtr = dst.ptr(i, j);

			for (int k = 0; k < src.chs; ++k) {

				if (ptr[k] < mint) {
					dstPtr[k] = 0;
				}
				else if (ptr[k] > maxt) {
					dstPtr[k] = 255;
				}
				else if ((i - 1 >= 0 && src.ptr(i - 1, j)[k] > maxt)                                    // up
					|| (j - 1 >= 0 && src.ptr(i, j - 1)[k] > maxt)                                      // left
					|| (j + 1 < src.cols && src.ptr(i, j + 1)[k] > maxt)                                // right
					|| (i + 1 < src.rows && src.ptr(i + 1, j)[k] > maxt)                                // down
					|| (i - 1 >=0  && j - 1 >= 0 && src.ptr(i - 1, j - 1)[k] > maxt)                    // up left
					|| (i - 1 >= 0 && j + 1 < src.cols && src.ptr(i - 1, j + 1)[k] > maxt)              // up right
					|| (i + 1 < src.rows && j + 1 < src.cols && src.ptr(i + 1, j + 1)[k] > maxt)        // down right
					|| (i + 1 < src.rows && j - 1 >= 0 && src.ptr(i + 1, j - 1)[k] > maxt)) {           // down left
					dstPtr[k] = 255;
				}
				else {
					dstPtr[k] = 0;
				}

			}
		}
	}
}

//////////////////////////////////////一阶微分算子///////////////////////////////////////////
// 重要问题：得到的边缘与灰度过度范围等宽，因此边缘可能无法被精确定位
//////////////////////////////////////一阶微分算子///////////////////////////////////////////

/**
 * @berif prewitt算子
 */
void prewitt(Matrix8u&src, Matrix8u&dst)
{
	if (!dst.equalSize(src))
		dst.create(src.rows, src.cols, src.chs);

	Matrix8s Gx(3, 3), Gy(3, 3);

	Gx = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
	Gy = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };

	int factor = 6;

	int *tempGx = new int[src.chs];
	int *tempGy = new int[src.chs];
	int *tempG = new int[src.chs];
	int zerosx = 0, zerosy = 0;
	unsigned char * srcPtr = nullptr;
	unsigned char * dstPtr = nullptr;

	for (int i = 0; i < dst.rows; ++i) {
		for (int j = 0; j < dst.cols; ++j) {

			memset(tempGx, 0, src.chs * sizeof(int));
			memset(tempGy, 0, src.chs * sizeof(int));
			memset(tempG, 0, src.chs * sizeof(int));
			zerosx = zerosy = 0;

			for (int ii = 0; ii < 3; ++ii) {
				for (int jj = 0; jj < 3; ++jj) {
                    auto _i = i - 1 + ii;
                    auto _j = j - 1 + jj;

					if (_i >=0  && _i < src.rows && _j >=0 && _j < src.cols) {
						for (int k = 0; k < src.chs; ++k) {
							tempGx[k] += src.ptr(_i, _j)[k] * Gx[ii][jj];
							tempGy[k] += src.ptr(_i, _j)[k] * Gy[ii][jj];
						}
					}
					else {
						zerosx += Gx[ii][jj];
						zerosy += Gy[ii][jj];
					}

				} // !for(jj)
			} // !for(ii)

			  // 局部梯度分量的的估计，通过给滤波结果乘以适当的尺度因子来实现
			for (int k = 0; k < src.chs; ++k) {
				if (zerosx != 0) {
					tempGx[k] /= zerosx;
				}
				else {
					tempGx[k] /= factor;
				}

				if (zerosy != 0) {
					tempGy[k] /= zerosy;
				}
				else {
					tempGy[k] /= factor;
				}
			}

			for (int k = 0; k < src.chs; ++k) {
                dst.ptr(i, j)[k] = (unsigned char)std::sqrt(tempGx[k] * tempGx[k] + tempGy[k] * tempGy[k]);
			}


		} // !for(j)
	} // !for(i)

	delete[] tempGx;
	delete[] tempGy;
	delete[] tempG;
}

void sobel(Matrix8u&src, Matrix8u&dst, int dx, int dy, int ksize)
{
	Matrix8u temp;
	sobel(src, dst, temp, dx, dy, ksize, true);
}

/**
 * @berif sobel算子
 * @param[in] src
 * @param[out] dst
 * @param[out] dstGD，
 * @param[in] ksize, must be 1, 3, 5 or 7.
 * @param[in] dx
 * @param[in] dy
 * @ksize[in] 卷积核的大小
 * @param[in] noGD，是否进行梯度非极大值抑制
 */
void sobel(Matrix8u&src, Matrix8u&dst, Matrix8u&dstGD, int dx, int dy, int ksize, bool noGD)
{
	if (!src.equalSize(dst))
		dst.create(src.rows, src.cols, src.chs);
	if (!noGD)
		if (!dstGD.equalSize(src))
			dstGD.create(src.rows, src.cols, src.chs);

	Matrix8s Gx(ksize, ksize), Gy(ksize, ksize);

	int factor = 0;

	switch (ksize) {
	case 1:
		
		break;

	case 3:
		// 原始sobel算子
		//Gx = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
		//Gy = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
		//factor = 8;
		// 改进型，可以将方向误差减到最小
		Gx = { -3, 0, 3, -10, 0, 10, -3, 0, 3 };
		Gy = { -3, -10, -3, 0, 0, 0, 3, 10, 3 };
		factor = 32;
		break;

	case 5:
		break;

	case 7:
		break;

	default:
		Z_Error("Error ksize!");
		return;
	}
	

	int *tempGx = new int[src.chs];
	int *tempGy = new int[src.chs];
	int *tempG = new int[src.chs];
	int zerosx = 0, zerosy = 0;
	int m = ksize / 2, n = ksize / 2;
	unsigned char * srcPtr = nullptr;
	unsigned char * dstPtr = nullptr;
	unsigned char * dstGDPtr = nullptr;
	int alpha = 0;
	double ang = 0;

	for (int i = 0; i < dst.rows; ++i) {
		for (int j = 0; j < dst.cols; ++j) {

			memset(tempGx, 0, src.chs * sizeof(int));
			memset(tempGy, 0, src.chs * sizeof(int));
			memset(tempG, 0, src.chs * sizeof(int));
			zerosx = zerosy = 0;

			for (int ii = 0; ii < ksize; ++ii) {
				for (int jj = 0; jj < ksize; ++jj) {
                    auto _i = i - m + ii;
                    auto _j = j - n + jj;
					if (_i >=0  && _i  < src.rows && _j >= 0 && _j < src.cols) {
						for (int k = 0; k < src.chs; ++k) {
							tempGx[k] += src.ptr(_i, _j)[k] * Gx[ii][jj];
							tempGy[k] += src.ptr(_i, _j)[k] * Gy[ii][jj];
						}
					}
					else {
						zerosx += Gx[ii][jj];
						zerosy += Gy[ii][jj];
					}

				} // !for(jj)
			} // !for(ii)

			// 局部梯度分量的的估计，通过给滤波结果乘以适当的尺度因子来实现
			for (int k = 0; k < src.chs; ++k) {
				if (zerosx != 0) {
					tempGx[k] /= zerosx;
				}
				else {
					tempGx[k] /= factor;
				}
				
				if (zerosy != 0) {
					tempGy[k] /= zerosy;
				}
				else {
					tempGy[k] /= factor;
				}
			}

			if (!noGD)
				dstGDPtr = dstGD.ptr(i, j);

			for (int k = 0; k < src.chs; ++k) {
                dst.ptr(i, j)[k] = (unsigned char)std::sqrt(tempGx[k] * tempGx[k] + tempGy[k] * tempGy[k]);
				// 计算梯度
				if (!noGD) {
					ang = atan2(tempGy[k],tempGx[k]) * RAD2ANG;

					if ((ang > -22.5 && ang < 22.5) || (ang > 157.5 || ang < -157.5))
						dstGDPtr[k] = 0;
					else if ((ang > 22.5 && ang < 67.5) || (ang < -112.5 && ang > -157.5))
						dstGDPtr[k] = 45;
					else if ((ang > 67.5 && ang < 112.5) || (ang < -67.5 && ang > -112.5))
						dstGDPtr[k] = 90;
					else if ((ang < -22.5 && ang > -67.5) || (ang > 112.5 && ang  < 157.5))
						dstGDPtr[k] = 135;
				}
			}


		} // !for(j)
	} // !for(i)

	delete[] tempGx;
	delete[] tempGy;
	delete[] tempG;
}




/**
 * @berif Canny 边缘检测算法
 *
 * @param[in] src，需要处理的图像
 * @param[out] dst，输出图像
 * @param[in] threshold1，双阈值第一个阈值
 * @param[in] threshold2，双阈值第二个阈值
 */
void Canny(Matrix8u&src, Matrix8u&dst, double threshold1, double threshold2, int apertureSize)
{
	// 中间变量
	Matrix8u dstGD;
	Matrix8u temp, temp1, temp2;

	// 第一步，高斯滤波
	GaussianBlur(src, temp, z::Size(5, 5));

	// 第二步，使用sobel算子
	sobel(temp, temp1, dstGD, 1, 1, apertureSize);

	// 第三步,非极大值抑制
	only_max(temp1, temp2, dstGD);
	
	// 第四步，双阈值
	double_threashold(temp2, dst, threshold1, threshold2);
}

void translation(Matrix8u &src, Matrix32s &kernel, Matrix8u &dst)
{
	if (!dst.equalSize(src)) {
		dst.create(src.rows, src.cols, src.chs);
	}
	dst.zeros();

	Matrix32s srcCoord(1,3,1), dstCoord;

	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			srcCoord = { i, j, 1 };
			dstCoord = srcCoord * kernel;

			for (int k = 0; k < src.chs && (dstCoord[0][0] < dst.rows && dstCoord[0][1] < dst.cols &&  dstCoord[0][0] >= 0 && dstCoord[0][1] >= 0); ++k) {
				dst.ptr(dstCoord[0][0], dstCoord[0][1])[k] = src.ptr(i, j)[k];
			}
		}
	}
}


#define _data(x, y) reinterpret_cast<char *>(src.ptr(x, y))[0]
void findContours(Matrix8u &src, std::vector<std::vector<Point>> &dst)
{
    std::vector<Point> middle_res;
    // 二进制化
    for (int i = src.rows * src.cols - 1; i >= 0; --i)
            if (src.data[i])
                src.data[i] = 1;
    
    int NBD = 1, LNBD = 1;
    Point p1, p2, p3, p4;

    // 边界上相邻两个点之间的相对位置
    uint8_t rpos;
    Point clockwise[8] = { { 0, -1 },{ -1, -1 },{ -1, 0 },{ -1, 1 },{ 0, 1 },{ 1, 1 },{ 1, 0 },{ 1, -1 } };      // 顺时针
    Point anticlockwise[8] = { { 0, -1 },{ 1, -1 },{ 1, 0 },{ 1, 1 },{ 0, 1 },{ -1, 1 },{ -1, 0 },{ -1, -1 } };  // 逆时针

    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {

            // step (1)
            // [ 0 | 1 ]
            if ((j - 1 < 0 || !_data(i, j - 1)) && _data(i, j) == 1) {
                NBD++;
                rpos = 0;

                if (NBD == 128) goto end;           // overflow

                p2 = { i, j - 1 };
            }
            // [ >=1 | 0 ]
            else if (_data(i, j) >= 1 && (j + 1 >= src.cols || !_data(i, j + 1))) {
                NBD++;
                rpos = 4;

                if (NBD == 128) goto end;           // overflow

                p2 = { i, j + 1 };
                if (_data(i, j) > 1)
                    LNBD = _data(i, j);
            }
            else {
                goto next;
            }

            // step (3.1)
            int k = 0;
            for (; k < 8; ++k) {
                p1 = clockwise[rpos++ & 0x07] + Point(i, j);
                if (p1.x >= 0 && p1.y >= 0 && p1.x < src.rows && p1.y < src.cols && _data(p1.x, p1.y))
                    break;
            }
            if (k == 8) {
                middle_res.push_back({ i, j });
                _data(i, j) = -NBD;
                dst.push_back(middle_res);
                middle_res.clear();
                goto next;
            }
            rpos = 8 - rpos + 1;

            // step (3.2)
            p2 = p1, p3 = { i, j };
            
            for (;;) {
                // step (3.3)
                for (int k = 0; k < 8; ++k) {
                    p4 = anticlockwise[++rpos & 0x07] + p3;
                    if (p4.x >= 0 && p4.y >= 0 && p4.x < src.rows && p4.y < src.cols && _data(p4.x, p4.y))
                        break;
                }
                rpos += 4;
               
                // step (3.4)
                if (p3.y + 1 >= src.cols || _data(p3.x, p3.y + 1) == 0) {
                    _data(p3.x, p3.y) = -NBD;
                    middle_res.push_back(p3);
                }
                else if (_data(p3.x, p3.y) == 1) {
                    _data(p3.x, p3.y) = NBD;
                    middle_res.push_back(p3);
                }   

                // step (3.5)
                if (p4.x == i && p4.y == j && p3 == p1) {
                    dst.push_back(middle_res);
                    middle_res.clear();
                    goto next;
                }
                else {
                    p2 = p3;
                    p3 = p4;
                }
            }

            // step (4)
        next:
            if (_data(i, j) != 1 && _data(i, j) != 0) {
                LNBD = abs(_data(i, j));
            }  
        }
    }
end:
    return;
}
#undef _data

#define _data(x, y) src.ptr(x, y)[0]
void findOutermostContours(Matrix8u &src, std::vector<std::vector<Point>> &dst)
{
    std::vector<Point> middle_res;
    int LNBD = 0;

    uint8_t rpos = 0;
    Point clockwise[8] = { { 0, -1 },{ -1, -1 },{ -1, 0 },{ -1, 1 },{ 0, 1 },{ 1, 1 },{ 1, 0 },{ 1, -1 } };      // 顺时针
    Point anticlockwise[8] = { { 0, -1 },{ 1, -1 },{ 1, 0 },{ 1, 1 },{ 0, 1 },{ -1, 1 },{ -1, 0 },{ -1, -1 } };  // 逆时针

    Point p1, p2, p3, p4;

    for (int i = 0; i < src.rows; ++i) {
        LNBD = 0;
        for (int j = 0; j < src.cols; ++j) {

            // [ 0 | 1] && LNBD == -2(254)
            if (((j - 1) <= 0 || !_data(i, j - 1)) && _data(i, j) == 255 && (LNBD == 0 || LNBD == 254))
                p2 = { i, j - 1 };
            else goto next;

            // 顺时针查找第一个点
            int k = 0;
            for (; k < 8; ++k) {
                p1 = clockwise[k] + Point(i, j);
                if (p1.x >= 0 && p1.y >= 0 && p1.x < src.rows && p1.y < src.cols && _data(p1.x, p1.y) != 0)
                    break;
            }
            if (k == 8) {
                middle_res.push_back({ i, j });
                _data(i, j) = 254;
                dst.push_back(middle_res);
                middle_res.clear();
                goto next;
            }
            rpos = 8 - k + 1;

            // step (3.2)
            p2 = p1, p3 = { i, j };
            for (;;) {
                // step (3.3)
                for (int k = 0; k < 8; ++k) {
                    p4 = anticlockwise[++rpos & 0x07] + p3;
                    if (p4.x >= 0 && p4.y >= 0 && p4.x < src.rows && p4.y < src.cols && _data(p4.x, p4.y))
                        break;
                }
                rpos += 4;

                // step (3.4)
                if (p3.y + 1 >= src.cols || _data(p3.x, p3.y + 1) == 0) {
                    _data(p3.x, p3.y) = 254;
                    middle_res.push_back(p3);
                }
                else if (_data(p3.x, p3.y) == 255) {
                    _data(p3.x, p3.y) = 2;
                    middle_res.push_back(p3);
                }

                // step (3.5)
                if (p4.x == i && p4.y == j && p3 == p1) {
                    dst.push_back(middle_res);
                    middle_res.clear();
                    goto next;
                }
                else {
                    p2 = p3;
                    p3 = p4;
                }
            }

        next:
            if (_data(i, j) != 0 && _data(i, j) != 255) {
                LNBD = _data(i, j);
            }
        }
    }
}
#undef _data

} // ! namespace z

