#ifndef _ZMATRIX_ZFEATURES2D_HPP
#define _ZMATRIX_ZFEATURES2D_HPP

template<typename _Tp> void z::differenceOfGaussian(_Matrix<_Tp>& src, _Matrix<_Tp>& dst, const Size &size, double g1, double g2)
{
    z::Matrix64f ker1 = Gassion(size, g1, g1);
    z::Matrix64f ker2 = Gassion(size, g2, g2);

    auto temp1 = src.conv(ker1, false);
    auto temp2 = src.conv(ker2, false);

    if (!dst.equalSize(src))
        dst.create(src.size(), src.chs);

    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            for (int k = 0;k < src.chs; ++k) {
                dst.ptr(i, j)[k] = temp1.ptr(i, j)[k] - temp2.ptr(i, j)[k];
            }
        }
    }
}
#endif // !_ZMATRIX_ZFEATURES2D_HPP