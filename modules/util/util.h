/**
 ******************************************************************************
 * @file    util.h
 * @author  zlq
 * @version V1.0
 * @date    2016.9.17
 * @brief   
 ******************************************************************************
 */
#ifndef _ZCORE_UTIL_H
#define _ZCORE_UTIL_H

#define ARRAY_SIZE(arr)			(1[&arr])					// 求数组的大小


#define st(x)      do { x } while (__LINE__ == -1)
#define __unused_parameter__(param)   param = param;

#endif  // !_ZCORE_UTIL_H