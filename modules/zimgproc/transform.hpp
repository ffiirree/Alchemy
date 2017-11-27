#ifndef _TRANSFORM_HPP
#define _TRANSFORM_HPP
#include <cassert>
#include "zcore/matrix.h"

namespace z
{
template <typename _Tp> void Laplacian(const _Matrix<_Tp>&src, _Matrix<_Tp>&dst, int ksize)
{
    assert(ksize > 0 && ksize % 2 == 1);

    Matrix8s kernel;
    if (ksize == 1) {
        kernel.create(3, 3);
        kernel = { 0, 1, 0, 1, -4, 1, 0, 1, 0 };
    }
    else if (ksize == 3) {
        kernel.create(3, 3);
        kernel = { 2, 0, 2, 0, -8, 0, 2, 0, 2 };
    }
    else {
        assert(1 == 0);
    }

    conv(src, dst, kernel);
}



/**
* @brief sobel算子
* @param[in] src
* @param[out] dst
* @param[out] dstGD
* @param[in] ksize Must be 1, 3, 5 or 7.
* @param[in] dx
* @param[in] dy
* @ksize[in] 卷积核的大小
* @param[in] noGD 是否进行梯度非极大值抑制
*/
template <typename _Tp>
void __sobel(_Matrix<_Tp>&src, _Matrix<_Tp>&dst, _Matrix<_Tp>&dstGD, int dx, int dy, int ksize, bool noGD, std::function<void(int&, int&)> callback)
{
    if (!src.equalSize(dst))
        dst.create(src.rows, src.cols, src.channels());
    if (!noGD)
    	if (!dstGD.equalSize(src))
    		dstGD.create(src.rows, src.cols, src.channels());

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
        Gx = {
            -3, 0,  3,
            -10, 0, 10,
            -3, 0,  3 };
        Gy = {
            -3, -10, -3,
            0,   0,  0,
            3,  10,  3 };

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


    int *tempGx = new int[src.channels()];
    int *tempGy = new int[src.channels()];
    int *tempG = new int[src.channels()];
    int m = ksize / 2, n = ksize / 2;
    unsigned char * dstGDPtr = nullptr;
    double ang = 0;

    for (int i = 0; i < dst.rows; ++i) {
        for (int j = 0; j < dst.cols; ++j) {

            memset(tempGx, 0, src.channels() * sizeof(int));
            memset(tempGy, 0, src.channels() * sizeof(int));
            memset(tempG, 0, src.channels() * sizeof(int));

            for (int ii = 0; ii < ksize; ++ii) {
                for (int jj = 0; jj < ksize; ++jj) {
                    auto _i = i - m + ii;
                    auto _j = j - n + jj;

                    if (!(static_cast<unsigned>(_i) < static_cast<unsigned>(dst.rows) &&
                        static_cast<unsigned>(_j) < static_cast<unsigned>(dst.cols))) {
                        callback(_i, _j);
                    }

                    for (int k = 0; k < src.channels(); ++k) {
                        tempGx[k] += src.at(_i, _j, k) * Gx[ii][jj];
                        tempGy[k] += src.at(_i, _j, k) * Gy[ii][jj];
                    }

                } // !for(jj)
            } // !for(ii)

            // 局部梯度分量的的估计，通过给滤波结果乘以适当的尺度因子来实现
            for (int k = 0; k < src.channels(); ++k) {
                tempGx[k] /= factor;
                tempGy[k] /= factor;
            }

              if (!noGD)
              	dstGDPtr = dstGD.ptr(i, j);

            for (int k = 0; k < src.channels(); ++k) {
                dst.at(i, j, k) = saturate_cast<uint8_t>(std::sqrt(tempGx[k] * tempGx[k] + tempGy[k] * tempGy[k]));
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

template <typename _Tp>
void Sobel(_Matrix<_Tp>&src, _Matrix<_Tp>&dst, int dx, int dy, int ksize, int borderType)
{
    Matrix8u temp;

    switch (borderType) {
        //!< `iiiiii|abcdefgh|iiiiiii`  with some specified `i`
    case BORDER_CONSTANT:
        //                                    break;
        //!< `aaaaaa|abcdefgh|hhhhhhh`
    case BORDER_REPLICATE:
        __sobel(src, dst, temp, dx, dy, ksize, true, BORDER_REPLICATE_CALLBACK(src));
        break;

        //!< `fedcba|abcdefgh|hgfedcb`
    case BORDER_REFLECT:
        __sobel(src, dst, temp, dx, dy, ksize, true, BORDER_REFLECT_CALLBACK(src));
        break;

        //!< `cdefgh|abcdefgh|abcdefg`
    case BORDER_WRAP:
        __sobel(src, dst, temp, dx, dy, ksize, true, BORDER_WRAP_CALLBACK(src));
        break;

        //!< `gfedcb|abcdefgh|gfedcba`
    default:
    case BORDER_REFLECT_101:
        __sobel(src, dst, temp, dx, dy, ksize, true, BORDER_DEFAULT_CALLBACK(src));
        break;

        //!< `uvwxyz|absdefgh|ijklmno`
    case BORDER_TRANSPARENT:
        _log_("Do not support!");
        break;
    }
}

}



#endif // !_TRANSFORM_H