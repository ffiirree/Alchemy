#include "zmath.h"


/**
 * @brief 求两个整数的均值
 * @attention 防止(x + y) / 2出现溢出
 */
int z::average(int x, int y)
{
	return (x & y) + ((x ^ y) >> 1);
}
