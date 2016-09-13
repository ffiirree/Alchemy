#ifndef _TRANSFORM_H
#define _TRANSFORM_H

#include "config_default.h"
#include "types_c.h"
#include "zmatrix.h"
#include "zimgproc.h"

#define RAD2ANG			((double)(57.296))

#ifdef __cplusplus
namespace z {


void sobel(Matrix8u&src, Matrix8u&dst, int dx, int dy, int ksize);
void sobel(Matrix8u&src, Matrix8u&dst, Matrix8u&dstGD, int dx = 1, int dy = 1, int ksize = 3, bool noGD = false);
void Canny(Matrix8u&src, Matrix8u&dst, double threshold1, double threshold2);




}


#endif

#include "transform.hpp"

#endif // !_TRANSFORM_H
