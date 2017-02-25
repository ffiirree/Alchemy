#ifndef _ZMATRIX_ZFEATURES2D_HPP
#define _ZMATRIX_ZFEATURES2D_HPP

template<typename _Tp> void z::differenceOfGaussian(_Matrix<_Tp>& src, _Matrix<_Tp>& dst, const Size &size, double g1, double g2)
{
    dst = src.conv(Gassion(size, g1, g1), true) - src.conv(Gassion(size, g2, g2), true);
}

#endif // !_ZMATRIX_ZFEATURES2D_HPP