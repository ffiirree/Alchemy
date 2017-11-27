/**
 ******************************************************************************
 * @file    transform.cpp
 * @author  zlq
 * @version V1.0
 * @date    2016.9.14
 * @brief   
 ******************************************************************************
 * @attention
 *
 *
 ******************************************************************************
 */
#include <cmath>
#include "transform.h"
#include "zcore/debug.h"
#include "zcore/util.h"
#include "zcore/types.h"

namespace z {

/**
 * @declaration 函数声明
 *              内部函数
 */ 
static void only_max(Matrix8u&src, Matrix8u&dst, Matrix8u&srcGD);
static void double_threashold(Matrix8u&src, Matrix8u&dst, double threshold1, double threshold2);
/**
 * @brief Canny中的非极大值抑制
 */
inline void only_max(Matrix8u&src, Matrix8u&dst, Matrix8u&srcGD)
{
    if (!src.equalSize(srcGD))
		throw std::runtime_error("src.equalSize(srcGD)!");

	dst = src.clone();
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {

		    auto srcptr = src.ptr(i, j);
		    auto dstptr = dst.ptr(i, j);

			for (int k = 0; k < src.channels(); ++k) {

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
                default: ;
                }
			} // for(k)
		} // for(j)
	} //  for(i)
}


/**
 * @brief Canny中的双阈值
 */
void double_threashold(Matrix8u&src, Matrix8u&dst, double threshold1, double threshold2)
{
	double maxt = threshold1 > threshold2 ? threshold1 : threshold2;
	double mint = threshold1 < threshold2 ? threshold1 : threshold2;

	if (!dst.equalSize(src))
		dst.create(src.rows, src.cols, src.channels());

    for (auto i = 0; i < src.rows; ++i) {
		for (auto j = 0; j < src.cols; ++j) {
		    auto ptr = src.ptr(i, j);
		    auto dstPtr = dst.ptr(i, j);

			for (auto k = 0; k < src.channels(); ++k) {

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
 * @brief prewitt算子
 */
void prewitt(Matrix8u&src, Matrix8u&dst)
{
	if (!dst.equalSize(src))
		dst.create(src.rows, src.cols, src.channels());

    Matrix8s Gx(3, 3);
    Matrix8s Gy(3, 3);

    Gx = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
	Gy = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };

	int factor = 6;

	int *tempGx = new int[src.channels()];
	int *tempGy = new int[src.channels()];
	int *tempG = new int[src.channels()];
	int zerosx = 0, zerosy = 0;
//	unsigned char * srcPtr = nullptr;
//	unsigned char * dstPtr = nullptr;

	for (int i = 0; i < dst.rows; ++i) {
		for (int j = 0; j < dst.cols; ++j) {

			memset(tempGx, 0, src.channels() * sizeof(int));
			memset(tempGy, 0, src.channels() * sizeof(int));
			memset(tempG, 0, src.channels() * sizeof(int));
			zerosx = zerosy = 0;

			for (int ii = 0; ii < 3; ++ii) {
				for (int jj = 0; jj < 3; ++jj) {
                    auto _i = i - 1 + ii;
                    auto _j = j - 1 + jj;

					if (_i >=0  && _i < src.rows && _j >=0 && _j < src.cols) {
						for (int k = 0; k < src.channels(); ++k) {
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
			for (int k = 0; k < src.channels(); ++k) {
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

			for (int k = 0; k < src.channels(); ++k) {
                dst.ptr(i, j)[k] = static_cast<uint8_t>(std::sqrt(tempGx[k] * tempGx[k] + tempGy[k] * tempGy[k]));
			}


		} // !for(j)
	} // !for(i)

	delete[] tempGx;
	delete[] tempGy;
	delete[] tempG;
}




/**
 * @brief Canny 边缘检测算法
 *
 * @param[in] src 需要处理的图像
 * @param[out] dst 输出图像
 * @param[in] threshold1 双阈值第一个阈值
 * @param[in] threshold2 双阈值第二个阈值
 */
void Canny(Matrix8u&src, Matrix8u&dst, double threshold1, double threshold2, int apertureSize)
{
	// 中间变量
	Matrix8u dstGD;
	Matrix8u temp, temp1, temp2;

	// 第一步，高斯滤波
	GaussianBlur(src, temp, z::Size(5, 5));

	// 第二步，使用sobel算子
	__sobel(temp, temp1, dstGD, 1, 1, apertureSize, false, BORDER_DEFAULT_CALLBACK(src));

	// 第三步,非极大值抑制
	only_max(temp1, temp2, dstGD);
	
	// 第四步，双阈值
	double_threashold(temp2, dst, threshold1, threshold2);
}

//void translation(Matrix8u &src, Matrix32s &kernel, Matrix8u &dst)
//{
//	//if (!dst.equalSize(src)) {
//	//	dst.create(src.rows, src.cols, src.channels());
//	//}
// //   dst = 0;
//
//	//Matrix32s srcCoord(1,3,1), dstCoord;
//
//	//for (int i = 0; i < src.rows; ++i) {
//	//	for (int j = 0; j < src.cols; ++j) {
// //           srcCoord = { i, j, 1 };
//	//		dstCoord = srcCoord * kernel;
//
//	//		for (int k = 0; k < src.channels() && (dstCoord[0][0] < dst.rows && dstCoord[0][1] < dst.cols &&  dstCoord[0][0] >= 0 && dstCoord[0][1] >= 0); ++k) {
//	//			dst.ptr(dstCoord[0][0], dstCoord[0][1])[k] = src.ptr(i, j)[k];
//	//		}
//	//	}
//	//}
//}


#define _data(x, y) src.at(x, y)
void findOutermostContours(Matrix8u &src, std::vector<std::vector<Point>> &dst)
{
    std::vector<Point> middle_res;

    // Binarization
    for(auto& p: src) 
        if (p) p = 1;
    
    
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

                if (NBD == 128) return;           // overflow

                p2 = { i, j - 1 };
            }
            // [ >=1 | 0 ]
            else if (_data(i, j) >= 1 && (j + 1 >= src.cols || !_data(i, j + 1))) {
                NBD++;
                rpos = 4;

                if (NBD == 128) return;           // overflow

                p2 = { i, j + 1 };
                if (_data(i, j) > 1)
                    LNBD = _data(i, j);
            }
            else {
                if (_data(i, j) != 1 && _data(i, j) != 0) {
                    LNBD = static_cast<int>(std::abs(_data(i, j)));
                }
                continue;
            }

            // step (3.1)
            int k = 0;
            for (; k < 8; ++k) {
                Point temp_(i, j);
                p1 = clockwise[rpos++ & 0x07] + temp_;
                if (p1.x >= 0 && p1.y >= 0 && p1.x < src.rows && p1.y < src.cols && _data(p1.x, p1.y))
                    break;
            }
            if (k == 8) {
                middle_res.push_back({ i, j });
                _data(i, j) = -NBD;
                dst.push_back(middle_res);
                middle_res.clear();
                
                if (_data(i, j) != 1 && _data(i, j) != 0) {
                    LNBD = static_cast<int>(std::abs(_data(i, j)));
                }
                continue;
            }
            rpos = 8 - rpos + 1;

            // step (3.2)
            p2 = p1, p3 = { i, j };
            
            for (;;) {
                // step (3.3)
                for (int k2 = 0; k2 < 8; ++k2) {
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
                    if (_data(i, j) != 1 && _data(i, j) != 0) {
                        LNBD = static_cast<int>(std::abs(_data(i, j)));
                    }
                    break;
                }
                else {
                    p2 = p3;
                    p3 = p4;
                }
            }

            // step (4)
//        next:
//            if (_data(i, j) != 1 && _data(i, j) != 0) {
//                LNBD = abs(_data(i, j));
//            }  
        }
    }
//end:
//    return;
}
#undef _data

#define _data(x, y) src.at(x, y)
void findContours(Matrix8u &src, std::vector<std::vector<Point>> &dst)
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
            else {
                if (_data(i, j) != 0 && _data(i, j) != 255) {
                    LNBD = _data(i, j);
                }
                continue;
            }

            // 顺时针查找第一个点
            uint8_t k = 0;
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
                
                if (_data(i, j) != 0 && _data(i, j) != 255) {
                    LNBD = _data(i, j);
                }
                continue;
            }
            rpos = 8 - k + 1;

            // step (3.2)
            p2 = p1, p3 = { i, j };
            for (;;) {
                // step (3.3)
                for (int kk = 0; kk < 8; ++kk) {
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
                    if (_data(i, j) != 0 && _data(i, j) != 255) {
                        LNBD = _data(i, j);
                    }
                    break;
                }
                else {
                    p2 = p3;
                    p3 = p4;
                }
            }
        }
    }
}
#undef _data

} // ! namespace z

