/**
 ******************************************************************************
 * @file    zmatch.cpp
 * @author  zlq
 * @version V1.0
 * @date    2016.9.14
 * @brief   简单的数学函数定义
 ******************************************************************************
 * @attention
 *
 *
 ******************************************************************************
 */
#ifndef _ZMATCH_H
#define _ZMATCH_H

#include "zcore\zmatrix.h"

template <typename _Tp> void max(_Tp *addr, size_t size, _Tp & max)
{
	max = *addr;
	_Tp * begin = addr + 1;
	_Tp *end = addr + size;

	for (; begin < end; ++begin) {
		if (max < begin[0])
			max = begin[0];
	}
}

template <typename _Tp> void min(_Tp *addr, size_t size, _Tp & min)
{
	min = *addr;
	_Tp * begin = addr + 1;
	_Tp *end = addr + size;

	for (; begin < end; ++begin) {
		if (min > begin[0])
			min = begin[0];
	}
}



#endif // !_ZMATCH_H
