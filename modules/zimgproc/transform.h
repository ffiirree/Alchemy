/**
 ******************************************************************************
 * @file    transform.h
 * @author  zlq
 * @version V1.0
 * @date    2016.9.14
 * @brief   图像变换的函数定义
 ******************************************************************************
 * @attention
 *
 *
 ******************************************************************************
 */
#ifndef _ZIMGPROC_TRANSFORM_H
#define _ZIMGPROC_TRANSFORM_H


#include "zcore/zmatrix.h"
#include "zimgproc.h"

#define RAD2ANG			((double)(57.296))

typedef enum {
    OUTER_BORDERS,
    ALL_BORDERS,
}ContourType;

namespace z {
// 图像几何变换
// 平移
void translation(Matrix8u &src, Matrix32s &kernel, Matrix8u &dst);

///////////////////////////////////////////////Edge Detector/////////////////////////////////////////////////
void getSobelKernels(Matrix8s &kx, Matrix8s &ky, int dx, int dy, int ksize);
template <typename _Tp> void Laplacian(const _Matrix<_Tp>&src, _Matrix<_Tp>&dst, int ksize = 1);
void prewitt(Matrix8u&src, Matrix8u&dst);
template <typename _Tp> void Sobel(_Matrix<_Tp>&src, _Matrix<_Tp>&dst, int dx = 1, int dy = 1, int ksize = 3, int borderType = BORDER_DEFAULT);
void Canny(Matrix8u&src, Matrix8u&dst, double threshold1, double threshold2, int apertureSize = 3);

///////////////////////////////////////////////Contours/////////////////////////////////////////////////
/**
 * @brief 提取所有轮廓
 * @attention 该函数会改变输入矩阵的值
 * @param[in/out] src 输入的二值图像
 * @param[out] dst 边界点集的向量
 * @ret None
 */
void findContours(Matrix8u &src, std::vector<std::vector<Point>> &dst);

/**
 * @brief 提取所有轮廓
 *      标记 [ 0 | 255 ] 为 2，[ 255 | 0 ]的为 254(-2)，
 *      如果扫描过2，说明在边界内，进过-2说明在边界外
 * @attention 该函数会改变输入矩阵的值
 * @param[in/out] src 输入的二值图像
 * @param[out] dst 边界点集的向量
 * @ret None
 */
void findOutermostContours(Matrix8u &src, std::vector<std::vector<Point>> &dst);
}

#include "transform.hpp"

#endif // !_ZIMGPROC_TRANSFORM_H
