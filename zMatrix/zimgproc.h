#ifndef _ZIMGPROC_H
#define _ZIMGPROC_H

#include <string>
#include "config_default.h"
#include "zmatrix.h"

#if defined(OPENCV)
#include <opencv2\core.hpp>
#endif

Matrix8u Mat2Matrix8u(cv::Mat & mat);

#endif