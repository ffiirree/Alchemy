#include "zmatch.h"


/**
 * @berif 求两个整数的均值
 * @ 黑科技，防止(x + y) / 2出现溢出
 */
int average(int x, int y)
{
	return (x & y) + ((x ^ y) >> 1);
}
