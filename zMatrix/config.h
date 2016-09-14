/**
 ******************************************************************************
 * @file    configt.h
 * @author  zlq
 * @version V1.0
 * @date    2016.9.7
 * @brief   添加调试信息，以及该库的一些默认设置
 ******************************************************************************
 * @attention
 *
 *
 ******************************************************************************
 */
#ifndef _CONFIG_DEFAULT_H
#define _CONFIG_DEFAULT_H

#include <Windows.h>
#include <iostream>
//#define DEBUG

#if defined(DEBUG)
#define _log_(str) do{\
	cout<<"[" << __FILE__ << "]" << "<" << __func__ << ">" << "(" << __LINE__ << ") : " << str << endl;\
}while (0)
#else
#define _log_(str)   
#endif

class TimeStamp {
public:
	inline void start() { GetLocalTime(&starttime); }
	void end();
	inline void runtime() 
	{
		GetLocalTime(&endtime);

		int diff = endtime.wMilliseconds - starttime.wMilliseconds;
		diff += (endtime.wSecond - starttime.wSecond) * 1000;
		diff += (endtime.wMinute - starttime.wMinute) * 60000;

		std::cout << "runtime = " << diff << "ms" << std::endl;
	}
private:
	SYSTEMTIME starttime, endtime;
};



// !是否使用了openCV
#ifndef OPENCV
#define OPENCV
#endif

#endif //!_CONFIG_DEFAULT_H

