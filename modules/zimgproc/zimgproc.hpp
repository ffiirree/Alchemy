/**
 ******************************************************************************
 * @file    zimgproc.hpp
 * @author  zlq
 * @version V1.0
 * @date    2016.9.14
 * @brief   图像处理相关模板函数的实现
 ******************************************************************************
 * @attention
 *
 *
 ******************************************************************************
 */
#ifndef _ZIMGPROC_HPP
#define _ZIMGPROC_HPP

#include <algorithm>
#include <vector>
#include "zcore/zdef.h"
#include "zcore/types.h"
#include "zmath/zmath.h"
#include "zcore/zmatrix.h"

namespace z {
template <class _Tp, int n>
bool operator==(const _Vec<_Tp, n>& v1, const _Vec<_Tp, n>& v2)
{
    for (auto i = 0; i < n; ++i) {
        if (v1[i] != v2[i]) {
            return false;
        }
    }
    return true;
}

template <class _Tp, int n>
bool operator!=(const _Vec<_Tp, n>& v1, const _Vec<_Tp, n>& v2)
{
    return !(v1 == v2);
}

template <class _Tp, int n>
std::ostream& operator<<(std::ostream& os, const _Vec<_Tp, n>& v)
{
    os << "[";
    if (sizeof(_Tp) == 1) {
        for (auto i = 0; i < n - 1; ++i) {
            os << static_cast<int>(v[i]) << ", ";
        }
        os << static_cast<int>(v[n - 1]) << "]";
    }
    else {
        for (auto i = 0; i < n - 1; ++i) {
            os << v[i] << ", ";
        }
        os << v[n - 1] << "]";
    }

    return os;
}

template<typename _Tp>
_Size<_Tp>& _Size<_Tp>::operator = (const _Size& sz)
{
    width = sz.width;
    height = sz.height;
    return *this;
}

template <class _Tp> void cvtColor(const _Matrix<_Tp>&src, _Matrix<_Tp>&dst, int code)
{
    auto is_hsv = false;

    switch (code) {
    case BGR2GRAY:
    {
        assert(src.channels() == 3);

        if (!(dst.size() == src.size() && dst.channels() == 1))
            dst.create(src.size(), 1);

        auto sbegin = src.template begin<_Vec<_Tp, 3>>();
        auto dbegin = dst.begin();
        for (; sbegin != src.template end<_Vec<_Tp, 3>>(); ++sbegin, ++dbegin) {
            auto pixel = *sbegin;
            *dbegin = saturate_cast<_Tp>(0.114 * pixel[0] + 0.587 * pixel[1] + 0.299 * pixel[2]);
        }
        break;
    }


    case BGR2RGB:
    {
        assert(src.channels() == 3);

        if (!dst.equalSize(src))
            dst.create(src.size(), src.channels());

        auto sbegin = src.template begin<_Vec<_Tp, 3>>();
        auto dbegin = dst.template begin<_Vec<_Tp, 3>>();
        for (; sbegin != src.template end<_Vec<_Tp, 3>>(); ++sbegin, ++dbegin) {
            (*dbegin)[0] = (*sbegin)[2];
            (*dbegin)[1] = (*sbegin)[1];
            (*dbegin)[2] = (*sbegin)[0];
        }
        break;
    }

    // 本hsv转换算法来自opencv官网:http://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html
    case BGR2HSV:
        is_hsv = true;
    case BGR2HSI:
    {
        assert(src.channels() == 3);

        if (!dst.equalSize(src))
            dst.create(src.size(), src.channels());

        for (int i = 0; i < src.rows; ++i) {
            for (int j = 0; j < src.cols; ++j) {
                auto spixel = src.template at<_Vec<_Tp, 3>>(i, j);
                auto&& dpixel = dst.template at<_Vec<_Tp, 3>>(i, j);

                _Tp _min, _max;
                double H = 0.0, S = 0.0;

                // min(R, G, B) & max(R, G, B)
                spixel[0] > spixel[1] ? (_max = spixel[0], _min = spixel[1]) : (_max = spixel[1], _min = spixel[0]);

                if (_max < spixel[2]) _max = spixel[2];
                if (_min > spixel[2]) _min = spixel[2];

                // V = max(R, G, B)
                if (is_hsv)
                    dpixel[2] = _max;
                else
                    dpixel[2] = _Tp((spixel[0] + spixel[1] + spixel[2]) / 3.0);

                // V != 0 ? S = (V - min(R,G,B))/V : S = 0;
                _max == 0 ? S = 0.0 : S = (_max - _min) / (double)_max;

                // if V == R : H = 60(G - B)/(V - min)
                // if V == G : H = 120 + 60(B - R)/(V - min)
                // if V == B : H = 240 + 60(R - G)/(V - min)
                if (_max == spixel[0]) {             // B
                    H = 240.0 + (60.0 * (spixel[2] - spixel[1])) / (_max - _min);
                }
                else if (_max == spixel[1]) {        // G
                    H = 120.0 + (60.0 * (spixel[0] - spixel[2])) / (_max - _min);
                }
                else if (_max == spixel[2]) {        // R
                    H = (60.0 * (spixel[1] - spixel[0])) / (_max - _min);
                }
                if (H < 0.0) H += 360;

                // 根据不同的深度进行处理
                if (sizeof(_Tp) == 1) {
                    dpixel[1] = _Tp(S * 255);
                    dpixel[0] = _Tp(H / 2);
                }
                else if (sizeof(_Tp) == 2) {
                    Z_Error("no support");
                }
                else if (sizeof(_Tp) == 4) {
                    Z_Error("no support");
                }
            }
        }

        break;
    }

    default:
        break;
    }
}

template <typename _T1, typename _T2>
void __conv(const _Matrix<_T1>& src, _Matrix<_T1>& dst, const _Matrix<_T2>& kernel, std::function<void(int&, int&)> callback)
{
    auto channels = src.channels();
    dst = _Matrix<_T1>::zeros(src.size(), channels);
    auto temp = new double[channels];
    auto temp_size = sizeof(double) * channels;

    for (auto i = 0; i < dst.rows; ++i) {
        for (auto j = 0; j < dst.cols; ++j) {

            memset(temp, 0, temp_size);

            for (auto ii = 0; ii < kernel.rows; ++ii) {
                for (auto jj = 0; jj < kernel.cols; ++jj) {

                    auto _i = i - kernel.rows / 2 + ii;
                    auto _j = j - kernel.cols / 2 + jj;

                    if (!(static_cast<unsigned>(_i) < static_cast<unsigned>(dst.rows) &&
                        static_cast<unsigned>(_j) < static_cast<unsigned>(dst.cols))) {
                        callback(_i, _j);
                    }

                    for (auto k = 0; k < channels; ++k) {
                        temp[k] += src.at(_i, _j, k) * kernel.at(ii, jj);
                    }
                }
            }

            for (auto k = 0; k < channels; ++k) {
                dst.at(i, j, k) = saturate_cast<_T1>(temp[k]);
            }
        }
    }

    delete[] temp;
}

template <typename _T1, typename _T2>
void conv(const _Matrix<_T1>& src, _Matrix<_T1>&dst, const _Matrix<_T2>& kernel, int borderType)
{
    assert(src.rows >= kernel.rows && src.cols >= kernel.cols);

    switch (borderType) {
        //!< `iiiiii|abcdefgh|iiiiiii`  with some specified `i`
    case BORDER_CONSTANT:
        //                                    break;
        //!< `aaaaaa|abcdefgh|hhhhhhh`
    case BORDER_REPLICATE:
        __conv(src, dst, kernel, BORDER_REPLICATE_CALLBACK(src));
        break;

        //!< `fedcba|abcdefgh|hgfedcb`
    case BORDER_REFLECT:
        __conv(src, dst, kernel, BORDER_REFLECT_CALLBACK(src));
        break;

        //!< `cdefgh|abcdefgh|abcdefg`
    case BORDER_WRAP:
        __conv(src, dst, kernel, BORDER_WRAP_CALLBACK(src));
        break;

        //!< `gfedcb|abcdefgh|gfedcba`
    default:
    case BORDER_REFLECT_101:
        __conv(src, dst, kernel, BORDER_DEFAULT_CALLBACK(src));
        break;

        //!< `uvwxyz|absdefgh|ijklmno`
    case BORDER_TRANSPARENT:
        _log_("Do not support!");
        break;
    }
}

template <typename _Tp> void blur(_Matrix<_Tp>& src, _Matrix<_Tp>& dst, Size size, int borderType)
{
    boxFilter(src, dst, size, true, borderType);
}


template <typename _Tp> void boxFilter(const _Matrix<_Tp>& src, _Matrix<_Tp>& dst, Size size, bool normalize, int borderType)
{
    assert(size.width == size.height || size.width % 2 != 0);

    auto kv = 1.0;
    if (normalize) kv = 1.0 / (size.width * size.height);

    Matrix64f kernel(size, 1, { kv });
    conv(src, dst, kernel);
}

template <typename _Tp> void GaussianBlur(_Matrix<_Tp>&src, _Matrix<_Tp> & dst, Size size, double sigmaX, double sigmaY, int borderType)
{
    conv(src, dst, Gassion(size, sigmaX, sigmaY), borderType);
}

template <typename _Tp> void embossingFilter(_Matrix<_Tp>& src, _Matrix<_Tp>&dst, Size size, int borderType)
{
    Matrix64f kernel(size);

    for (int i = 0; i < kernel.rows; ++i) {
        for (int j = 0; j < kernel.cols; ++j) {
            if (j < kernel.rows - i - 1)
                kernel[i][j] = -1;
            else if (j > kernel.rows - i - 1)
                kernel[i][j] = 1;
            else
                kernel[i][j] = 0;
        }
    }
    conv(src, dst, kernel, borderType);
}

template <typename _Tp> void __medianFilter(_Matrix<_Tp>&src, _Matrix<_Tp>& dst, Size size, std::function<void(int &i, int& j)> callback)
{
    auto area = size.area();
    _Matrix<_Tp> buffer(src.channels(), area);

    if (!src.equalSize(dst))
        dst.create(src.size(), src.channels());

    auto m = size.width / 2, n = size.height / 2;
    int valindex = area / 2;

    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {

            for (int ii = 0; ii < size.width; ++ii) {
                for (int jj = 0; jj < size.height; ++jj) {
                    auto _i = i - m + ii;
                    auto _j = j - n + jj;

                    for (auto k = 0; k < src.channels(); ++k) {
                        if (!(static_cast<unsigned>(_i) < static_cast<unsigned>(dst.rows) &&
                            static_cast<unsigned>(_j) < static_cast<unsigned>(dst.cols))) {
                            callback(_i, _j);
                        }
                        buffer.at(k, ii * size.width + jj) = src.at(_i, _j, k);
                    }
                }
            }
            for (auto k = 0; k < src.channels(); ++k) {
                std::sort(buffer.ptr(k), buffer.ptr(k) + area);  // 占95%以上的时间
                dst.at(i, j, k) = buffer.at(k, valindex);
            }

        } // !for(j)
    } // !for(i)
}

template <typename _Tp> void medianFilter(_Matrix<_Tp>&src, _Matrix<_Tp>& dst, Size size, int borderType)
{
    switch (borderType) {
        //!< `iiiiii|abcdefgh|iiiiiii`  with some specified `i`
    case BORDER_CONSTANT:
        //                                    break;
        //!< `aaaaaa|abcdefgh|hhhhhhh`
    case BORDER_REPLICATE:
        __medianFilter(src, dst, size, BORDER_REPLICATE_CALLBACK(src));
        break;

        //!< `fedcba|abcdefgh|hgfedcb`
    case BORDER_REFLECT:
        __medianFilter(src, dst, size, BORDER_REFLECT_CALLBACK(src));
        break;

        //!< `cdefgh|abcdefgh|abcdefg`
    case BORDER_WRAP:
        __medianFilter(src, dst, size, BORDER_WRAP_CALLBACK(src));
        break;

        //!< `gfedcb|abcdefgh|gfedcba`
    default:
    case BORDER_REFLECT_101:
        __medianFilter(src, dst, size, BORDER_DEFAULT_CALLBACK(src));
        break;

        //!< `uvwxyz|absdefgh|ijklmno`
    case BORDER_TRANSPARENT:
        _log_("Do not support!");
        break;
    }
}

// attention: pix: 1*8 or 3*8 uchar
template <typename _Tp> void bilateralFilter(const _Matrix<_Tp>&src, _Matrix<_Tp>&dst, int d, double sigmaColor, double sigmaSpace)
{
    if (!dst.equalSize(src))
        dst.create(src.size(), src.channels());

    auto r = 0, max_ofs = 0;
    //
    if (sigmaColor <= 0) sigmaColor = 1;
    if (sigmaSpace <= 0) sigmaSpace = 1;

    auto gauss_color_coeff = -0.5 / (sigmaColor * sigmaColor);
    auto gauss_space_coeff = -0.5 / (sigmaSpace * sigmaSpace);

    if (d < 0) r = static_cast<int>(sigmaSpace * 1.5);
    else r = d / 2;

    d = r * 2 + 1;

    // 牺牲存储来换取时间
    double * color_weight = new double[src.chs * 256];
    double * space_weight = new double[d * d];
    int * space_ofs = new int[d * d];

    // initialize color-related bilateral filter coifficients
    for (int i = 0; i < src.chs * 256; ++i)
        color_weight[i] = std::exp(i * i * gauss_color_coeff);

    for (int i = -r; i <= r; ++i) {
        for (int j = -r; j <= r; ++j) {
            auto r_t = std::sqrt(static_cast<double>(i) * i + static_cast<double>(j) * j);
            if (r_t <= r) {
                space_weight[max_ofs] = std::exp(r_t * r_t * gauss_space_coeff);
                space_ofs[max_ofs++] = static_cast<int>(i * src.step + j * src.channels());
            }
        }
    }

    double *temp_val = new double[src.channels()];

    auto ptr = src.ptr();
    auto data_len = src.size_ * src.channels();

    for (int i = 0; i < data_len; i += src.channels()) {
        double norm = 0;
        int mv = 0;
        for (int k = 0; k < src.channels(); ++k) {
            mv += ptr[i + k];
        }

        memset(temp_val, 0, sizeof(double) * src.channels());//清零

        for (int j = 0; j < max_ofs; ++j) {
            double w1 = space_weight[j];

            int cv = 0;
            int c_pos = i + space_ofs[j];
            if ((unsigned)c_pos < (unsigned)data_len) {
                for (int k = 0; k < src.channels(); ++k) {
                    cv += ptr[c_pos + k];
                }

                double w2 = color_weight[abs(cv - mv)];
                double w = w1 * w2;
                norm += w;
                for (int k = 0; k < src.channels(); ++k) {
                    temp_val[k] += ptr[c_pos + k] * w;
                }
            }
        }
        for (int k = 0; k < src.channels(); ++k) {
            dst.data[i + k] = saturate_cast<_Tp>(temp_val[k] / norm);
        }
    }

    delete[] color_weight;
    delete[] space_weight;
    delete[] space_ofs;
    delete[] temp_val;
}

//////////////////////////////////////形态学滤波//////////////////////////////////////
template <typename _Tp> void __morphOp(int code, _Matrix<_Tp>& src, _Matrix<_Tp>&dst, Size size, std::function<void(int&, int&)> callback)
{
    auto area = size.area();
    _Matrix<_Tp> buffer(src.channels(), area);

    if (!src.equalSize(dst))
        dst.create(src.rows, src.cols, src.channels());

    for (auto i = 0; i < src.rows; ++i) {
        for (auto j = 0; j < src.cols; ++j) {

            for (auto ii = 0; ii < size.width; ++ii) {
                for (auto jj = 0; jj < size.height; ++jj) {
                    auto _i = i - size.width / 2 + ii;
                    auto _j = j - size.height / 2 + jj;

                    for (auto k = 0; k < src.channels(); ++k) {
                        if (!(static_cast<unsigned>(_i) < static_cast<unsigned>(dst.rows) &&
                            static_cast<unsigned>(_j) < static_cast<unsigned>(dst.cols))) {
                            callback(_i, _j);
                        }

                        buffer.at(k, ii * size.width + jj) = src.at(_i, _j, k);
                    }
                }
            }
            switch (code) {
            case MORP_ERODE:
                for (auto k = 0; k < src.channels(); ++k) {
                    _Tp minVal = 0;
                    z::min(buffer.ptr(k), area, minVal);
                    dst.at(i, j, k) = minVal;
                }
                break;

            case MORP_DILATE:
                for (auto k = 0; k < src.channels(); ++k) {
                    _Tp maxVal = 0;
                    z::max(buffer.ptr(k), area, maxVal);
                    dst.at(i, j, k) = maxVal;
                }
                break;

            default: break;
            }
        } // !for(j)
    } // !for(i)
}
template <typename _Tp> void morphOp(int code, _Matrix<_Tp>& src, _Matrix<_Tp>&dst, Size size, int borderType)
{
    switch (borderType) {
        //!< `iiiiii|abcdefgh|iiiiiii`  with some specified `i`
    case BORDER_CONSTANT:
        //                                    break;
        //!< `aaaaaa|abcdefgh|hhhhhhh`
    case BORDER_REPLICATE:
        __morphOp(code, src, dst, size, BORDER_REPLICATE_CALLBACK(src));
        break;

        //!< `fedcba|abcdefgh|hgfedcb`
    case BORDER_REFLECT:
        __morphOp(code, src, dst, size, BORDER_REFLECT_CALLBACK(src));
        break;

        //!< `cdefgh|abcdefgh|abcdefg`
    case BORDER_WRAP:
        __morphOp(code, src, dst, size, BORDER_WRAP_CALLBACK(src));
        break;

        //!< `gfedcb|abcdefgh|gfedcba`
    default:
    case BORDER_REFLECT_101:
        __morphOp(code, src, dst, size, BORDER_DEFAULT_CALLBACK(src));
        break;

        //!< `uvwxyz|absdefgh|ijklmno`
    case BORDER_TRANSPARENT:
        _log_("Do not support!");
        break;
    }
}

template <typename _Tp> void erode(_Matrix<_Tp>& src, _Matrix<_Tp>&dst, Size kernel, int borderType)
{
    morphOp(MORP_ERODE, src, dst, kernel, borderType);
}

template <typename _Tp> void dilate(_Matrix<_Tp>& src, _Matrix<_Tp>&dst, Size kernel, int borderType)
{
    morphOp(MORP_DILATE, src, dst, kernel, borderType);
}

template <typename _Tp> void open(_Matrix<_Tp>& src, _Matrix<_Tp>&dst, Size kernel, int borderType)
{
    _Matrix<_Tp> _dst;
    morphOp(MORP_ERODE, src, _dst, kernel, borderType);
    morphOp(MORP_DILATE, _dst, dst, kernel, borderType);
}

template <typename _Tp> void morphEx(_Matrix<_Tp>& src, _Matrix<_Tp>&dst, int op, Size kernel, int borderType)
{
    _Matrix<_Tp> temp;
    if (dst.equalSize(src))
        dst.create(src.rows, src.cols, src.channels());

    switch (op) {
    case MORP_ERODE:
        erode(src, dst, kernel, borderType);
        break;

    case MORP_DILATE:
        dilate(src, dst, kernel, borderType);
        break;

    case MORP_OPEN:
        erode(src, temp, kernel, borderType);
        dilate(temp, dst, kernel, borderType);
        break;

    case MORP_CLOSE:
        dilate(src, temp, kernel, borderType);
        erode(temp, dst, kernel, borderType);
        break;

    case MORP_BLACKHAT:
        dilate(src, temp, kernel, borderType);
        erode(temp, dst, kernel, borderType);

        dst -= src;
        break;

    case MORP_TOPHAT:
        erode(src, temp, kernel, borderType);
        dilate(temp, dst, kernel, borderType);

        dst = src - dst;
        break;

    case MORP_GRADIENT:
        erode(src, temp, kernel, borderType);
        dilate(src, dst, kernel, borderType);

        dst -= temp;
        break;
    default:;
    }
}

template <typename _Tp> void spilt(const _Matrix<_Tp> & src, std::vector<_Matrix<_Tp>> & mv)
{
    mv = std::vector<_Matrix<_Tp>>(src.channels());

    for (auto i = 0; i < src.channels(); ++i) {
        mv.at(i).create(src.rows, src.cols, 1);
    }

    for (auto i = 0; i < src.rows; ++i) {
        for (auto j = 0; j < src.cols; ++j) {
            for (auto k = 0; k < src.channels(); ++k) {
                mv.at(k).at(i, j) = src.ptr(i, j)[k];
            }
        }
    }
}

template <typename _Tp> void merge(const _Matrix<_Tp> & src1, const _Matrix<_Tp> & src2, _Matrix<_Tp> & dst)
{
    if (!src1.equalSize(src2))
        _log_("!src1.equalSize(src2)");

    if (dst.size() != src1.size())
        dst.create(src1.rows, src1.cols, 2);

    for (auto i = 0; i < src1.rows; ++i) {
        for (auto j = 0; j < src2.cols; ++j) {
            dst.at(i, j, 0) = src1.at(i, j);
            dst.at(i, j, 1) = src2.at(i, j);
        }
    }
}

template <typename _Tp> void merge(const std::vector<_Matrix<_Tp>> & src, _Matrix<_Tp> & dst)
{
    assert(src.size() >= 1);
    
#ifndef NDEBUG
    for(const auto & mat: src) {
        assert(mat.size() == src.at(0).size() && mat.channels() == 1);
    }
#endif

    int rows = src.at(0).rows;
    int cols = src.at(0).cols;
    int chs = src.size();

    // Alloc the memory.
    dst.create(rows, cols, chs);

    // merge
    for (auto i = 0; i < rows; ++i) {
        for (auto j = 0; j < cols; ++j) {
            for (auto k = 0; k < chs; ++k) {
                dst.at(i, j, k) = src.at(k).at(i, j);
            }
        }
    }
}

template <typename _Tp> 
void copyMakeBorder(const _Matrix<_Tp> & src, _Matrix<_Tp> & dst, int top, int bottom, int left, int right)
{
    dst = _Matrix<_Tp>::zeros(src.rows + top + bottom, src.cols + left + right, src.channels());

    for (auto i = top; i < src.rows + top; ++i) {
        for (auto j = left; j < src.cols + left; ++j) {
            for (auto k = 0; k < dst.channels(); ++k) {
                dst.at(i, j, k) = src.at(i - top, j - left, k);
            }
        }
    }
}


template <typename _Tp>
void threshold(_Matrix<_Tp> &src, _Matrix<_Tp>& dst, double thresh, double maxval, int type)
{
    assert(src.channels() == 1);
    dst = src.clone();

    switch (type) {
    case THRESH_BINARY:
        for(auto & pixel : dst) {
            pixel > saturate_cast<_Tp>(thresh) ? pixel = saturate_cast<_Tp>(maxval) : pixel = 0;
        }
        break;

    case THRESH_BINARY_INV:
        for (auto & pixel : dst) {
            pixel < saturate_cast<_Tp>(thresh) ? pixel = saturate_cast<_Tp>(maxval) : pixel = 0;
        }
        break;

    case THRESH_TRUNC:
        for (auto & pixel : dst) {
            if(pixel > saturate_cast<_Tp>(thresh)) {
                pixel = saturate_cast<_Tp>(thresh);
            }
        }
        break;

    case THRESH_TOZERO:
        for (auto & pixel : dst) {
            if(pixel <= saturate_cast<_Tp>(thresh)) {
                pixel = 0;
            }
        }
        break;

    case THRESH_TOZERO_INV:
        for (auto & pixel : dst) {
            if (pixel > saturate_cast<_Tp>(thresh)) {
                pixel = 0;
            }
        }
        break;

    default:break;
    }
}


template <typename _Tp> void pyrUp(const _Matrix<_Tp>& src, _Matrix<_Tp>& dst)
{
    auto kernel = Gassion({ 5, 5 }, 0, 0);
    kernel = kernel * 4;

    auto temp = _Matrix<_Tp>::zeros(src.rows * 2, src.cols * 2, src.channels());

    for (auto i = 0; i < src.rows; ++i)
        for (auto j = 0; j < src.cols; ++j)
            for (auto k = 0; k < src.channels(); ++k)
                temp.at(2 * i, 2 * j, k) = src.at(i, j, k);

    conv(temp, dst, kernel);
}


template <typename _Tp> void pyrDown(const _Matrix<_Tp>& src, _Matrix<_Tp>& dst)
{
    conv(src, dst, Gassion({ 5, 5 }, 0, 0));

    int dst_rows = src.rows / 2;
    int dst_cols = src.cols / 2;

    dst.create(dst_rows, dst_cols, src.channels());
    for (auto i = 0; i < dst_rows; ++i)
        for (auto j = 0; j < dst_cols; ++j)
            for (auto k = 0; k < src.channels(); ++k)
                dst.at(i, j, k) = src.at(2 * i, 2 * j, k);
}
};

#endif // !_ZIMGPROC_HPP 