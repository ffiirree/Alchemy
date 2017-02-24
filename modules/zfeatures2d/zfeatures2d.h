#ifndef _ZMATRIX_ZFEATURES2D_H
#define _ZMATRIX_ZFEATURES2D_H

#include "zcore\zmatrix.h"
#include "zimgproc\zimgproc.h"

namespace z {
    template<typename _Tp> void differenceOfGaussian(_Matrix<_Tp>& src, _Matrix<_Tp>& dst, const Size &size, double g1, double g2);
};


#include "zfeatures2d.hpp"
#endif