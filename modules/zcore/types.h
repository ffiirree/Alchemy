/**
 ******************************************************************************
 * @file    types.cpp
 * @author  zlq
 * @version V1.0
 * @date    2016.9.14
 * @brief   主要是各种枚举类型的定义
 ******************************************************************************
 * @attention
 *
 *
 ******************************************************************************
 */
#ifndef _TYPES_C_H
#define _TYPES_C_H

/**
 * \ 颜色空间转换
 */
enum
{
	BGR2GRAY = 0,
	BGR2RGB,
    BGR2HSV,
    BGR2HSI,
};

/**
 * \ 形态学滤波方式
 */
enum {
	MORP_ERODE = 0,
	MORP_DILATE,
	MORP_OPEN,
	MORP_CLOSE,
	MORP_TOPHAT,
	MORP_BLACKHAT,
	MORP_GRADIENT
};

/**
 * \ 单通道固定阈值方式
 */
enum {
    THRESH_BINARY,
    THRESH_BINARY_INV,
    THRESH_TRUNC,
    THRESH_TOZERO,
    THRESH_TOZERO_INV,
};
#endif // !_TYPES_C_H