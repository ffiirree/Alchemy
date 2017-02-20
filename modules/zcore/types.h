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

// Z_XX & 0110b
#define Z_8U        0           // 1: 0000 -> 0000 : 0
#define Z_8S        1           // 1: 0001 -> 0000 : 0
#define Z_16U       2           // 2: 0010 -> 0010 : 2
#define Z_16S       3           // 2: 0011 -> 0010 : 4
#define Z_32S       4           // 4: 0100 -> 0100 : 6
#define Z_32F       5           // 4: 0101 -> 0100 : 6
#define Z_64F       6           // 8: 0110 -> 0100 : 6

#define Z_MAKE_TYPE(d, n)       ((d & 0x06) | (n << 4))

#define Z_8UC1      Z_MAKE_TYPE(Z_8U, 1)
#define Z_8UC2      Z_MAKE_TYPE(Z_8U, 2)
#define Z_8UC3      Z_MAKE_TYPE(Z_8U, 3)
#define Z_8UC4      Z_MAKE_TYPE(Z_8U, 4)
#define Z_8UC(n)    Z_MAKE_TYPE(Z_8U, (n))

#define Z_8SC1      Z_MAKE_TYPE(Z_8S, 1)
#define Z_8SC2      Z_MAKE_TYPE(Z_8S, 2)
#define Z_8SC3      Z_MAKE_TYPE(Z_8S, 3)
#define Z_8SC4      Z_MAKE_TYPE(Z_8S, 4)
#define Z_8SC(n)    Z_MAKE_TYPE(Z_8S, (n))

#define Z_16UC1     Z_MAKE_TYPE(Z_16U, 1)
#define Z_16UC2     Z_MAKE_TYPE(Z_16U, 2)
#define Z_16UC3     Z_MAKE_TYPE(Z_16U, 3)
#define Z_16UC4     Z_MAKE_TYPE(Z_16U, 4)
#define Z_16UC(n)   Z_MAKE_TYPE(Z_16U, (n))

#define Z_16SC1     Z_MAKE_TYPE(Z_16S, 1)
#define Z_16SC2     Z_MAKE_TYPE(Z_16S, 2)
#define Z_16SC3     Z_MAKE_TYPE(Z_16S, 3)
#define Z_16SC4     Z_MAKE_TYPE(Z_16S, 4)
#define Z_16SC(n)   Z_MAKE_TYPE(Z_16S, (n))

#define Z_32SC1     Z_MAKE_TYPE(Z_32S, 1)
#define Z_32SC2     Z_MAKE_TYPE(Z_32S, 2)
#define Z_32SC3     Z_MAKE_TYPE(Z_32S, 3)
#define Z_32SC4     Z_MAKE_TYPE(Z_32S, 4)
#define Z_32SC(n)   Z_MAKE_TYPE(Z_32S, (n))

#define Z_32FC1     Z_MAKE_TYPE(Z_32F, 1)
#define Z_32FC2     Z_MAKE_TYPE(Z_32F, 2)
#define Z_32FC3     Z_MAKE_TYPE(Z_32F, 3)
#define Z_32FC4     Z_MAKE_TYPE(Z_32F, 4)
#define Z_32FC(n)   Z_MAKE_TYPE(Z_32F, (n))

#define Z_64FC1     Z_MAKE_TYPE(Z_64F, 1)
#define Z_64FC2     Z_MAKE_TYPE(Z_64F, 2)
#define Z_64FC3     Z_MAKE_TYPE(Z_64F, 3)
#define Z_64FC4     Z_MAKE_TYPE(Z_64F, 4)
#define Z_64FC(n)   Z_MAKE_TYPE(Z_64F, (n))

enum
{
	BGR2GRAY = 0,
	BGR2RGB
};


enum {
	MORP_ERODE = 0,
	MORP_DILATE,
	MORP_OPEN,
	MORP_CLOSE,
	MORP_TOPHAT,
	MORP_BLACKHAT,
	MORP_GRADIENT
};
#endif // !_TYPES_C_H