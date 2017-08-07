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
#include "zcore/zmatrix.h"
#include "zcore/types.h"

//#include <cmath>

namespace z {

	template<typename _Tp> inline _Size<_Tp>& _Size<_Tp>::operator = (const _Size& sz)
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
            assert(src.chs == 3);

			if (!(dst.rows == src.rows && dst.cols == src.cols && dst.chs == 1 && src.chs == 3))
				dst.create(src.size(), 1);

			const _Tp * srcPtr = nullptr;

			for (auto i = 0; i < src.rows; ++i) {
				for (auto j = 0; j < src.cols; ++j) {

					srcPtr = src.ptr(i, j);

					dst.ptr(i, j)[0] = _Tp(0.114 * srcPtr[0] + 0.587 * srcPtr[1] + 0.299 * srcPtr[2]);
				}
			}
            break;
		}
			

		case BGR2RGB:
        {
            assert(src.chs == 3);

            if (!dst.equalSize(src)) {
                dst.create(src.size(), src.chs);
            }

            for (auto i = 0; i < src.rows; ++i) {
                for (auto j = 0; j < src.cols; ++j) {
                    dst.ptr(i, j)[2] = src.ptr(i, j)[0];
                    dst.ptr(i, j)[1] = src.ptr(i, j)[1];
                    dst.ptr(i, j)[0] = src.ptr(i, j)[2];
                }
            }
            break;
        }
        
        // 本hsv转换算法来自opencv官网:http://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html
        case BGR2HSV:
            is_hsv = true;
        case BGR2HSI:
        {
            assert(src.chs == 3);

            if (!dst.equalSize(src))
                dst.create(src.size(), src.chs);

            for (int i = 0; i < src.rows; ++i) {
                for (int j = 0; j < src.cols; ++j) {
                    const Vec3u8* src_p = src.template ptr<Vec3u8>(i, j);
                    Vec3u8* dst_p = dst.template ptr<Vec3u8>(i, j);

                    _Tp _min, _max;
                    double H = 0.0, S = 0.0;

                    // min(R, G, B) & max(R, G, B)
                    (*src_p)[0] > (*src_p)[1] ? (_max = (*src_p)[0], _min = (*src_p)[1]) : (_max = (*src_p)[1], _min = (*src_p)[0]);
                    
                    if (_max < (*src_p)[2]) _max = (*src_p)[2];
                    if (_min > (*src_p)[2]) _min = (*src_p)[2];

                    // V = max(R, G, B)
                    if (is_hsv)
                        (*dst_p)[2] = _max;
                    else
                        (*dst_p)[2] = _Tp(((*src_p)[0] + (*src_p)[1] + (*src_p)[2]) / 3.0);

                    // V != 0 ? S = (V - min(R,G,B))/V : S = 0;
                    _max == 0 ? S = 0.0 : S = (_max - _min) / (double)_max;

                    // if V == R : H = 60(G - B)/(V - min)
                    // if V == G : H = 120 + 60(B - R)/(V - min)
                    // if V == B : H = 240 + 60(R - G)/(V - min)
                    if (_max == (*src_p)[0]) {             // B
                        H = 240.0 + (60.0 * ((*src_p)[2] - (*src_p)[1])) / (_max - _min);
                    }
                    else if (_max == (*src_p)[1]) {        // G
                        H = 120.0 + (60.0 * ((*src_p)[0] - (*src_p)[2])) / (_max - _min);
                    }
                    else if (_max == (*src_p)[2]) {        // R
                        H = (60.0 * ((*src_p)[1] - (*src_p)[0])) / (_max - _min);
                    }
                    if (H < 0.0) H += 360;

                    // 根据不同的深度进行处理
                    if (sizeof(_Tp) == 1) {
                        (*dst_p)[1] = _Tp(S * 255);
                        (*dst_p)[0] = _Tp(H / 2);
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

	/**
	 * @brief 均值滤波
	 */
	template <typename _Tp> void blur(_Matrix<_Tp>& src, _Matrix<_Tp>& dst, Size size)
	{
		boxFilter(src, dst, size, true);
	}

	/**
	 * @brief 方框滤波
	 * @param[in] normalize，是否归一化，卷积核各项和不为1时除以和。
	 */
	template <typename _Tp> void boxFilter(const _Matrix<_Tp>& src, _Matrix<_Tp>& dst, Size size, bool normalize)
	{
        assert(size.width == size.height || size.width % 2 != 0);

        z::Matrix64f kernel(size);
        kernel.ones();
        src.conv(kernel, dst, normalize);
	}

	/**
	 * @brief 高斯滤波
	 * @param[in] normalize，是否归一化，卷积核各项和不为1时除以和。
	 */
	template <typename _Tp> void GaussianBlur(_Matrix<_Tp>&src, _Matrix<_Tp> & dst, Size size, double sigmaX, double sigmaY)
	{
        Matrix64f kernel = Gassion(size, sigmaX, sigmaY);
        dst = src.conv(kernel, false);
	}

	template <typename _Tp> void embossingFilter(_Matrix<_Tp>& src, _Matrix<_Tp>&dst, Size size)
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
		src.conv(kernel, dst, false);
	}
	
	template <typename _Tp> void medianFilter(_Matrix<_Tp>&src, _Matrix<_Tp>& dst, Size size)
	{
		int area = size.area();
		_Tp ** ker = new _Tp *[src.chs];
		for (int i = 0; i < src.chs; ++i) {
			ker[i] = new _Tp[area];
		}

		if (!src.equalSize(dst))
			dst.create(src.rows, src.cols, src.chs);

		int m = size.width / 2, n = size.height / 2;
		int cnt = 0;
		int valindex = 0;
		int valDefault = area / 2;

		for (int i = 0; i < src.rows; ++i) {
			for (int j = 0; j < src.cols; ++j) {

				cnt = 0;
				for (int ii = 0; ii < size.width; ++ii) {
					for (int jj = 0; jj < size.height; ++jj) {
                        auto _i = i - m + ii;
                        auto _j = j - n + jj;
						if (_i >= 0  && _i < src.rows && _j >= 0 && _j < src.cols) {
							for (int k = 0; k < src.chs; ++k) {
								ker[k][cnt] = src.ptr(_i, _j)[k];
							}
							cnt++;
						}
					}
				}
                cnt != area ? (valindex = cnt / 2) : (valindex = valDefault);
				for (int k = 0; k < src.chs; ++k) {
					std::sort(ker[k], ker[k] + cnt);  // 占95%以上的时间
                    dst.ptr(i, j)[k] = ker[k][valindex];
				}

			} // !for(j)
		} // !for(i)

		for (int i = 0; i < src.chs; ++i) {
			delete[] ker[i];
		}
		delete[] ker;
	}

    // attention: pix: 1*8 or 3*8 uchar
    template <typename _Tp> void bilateralFilter(const _Matrix<_Tp>&src, _Matrix<_Tp>&dst, int d, double sigmaColor, double sigmaSpace)
    {
        if (!dst.equalSize(src))
            dst.create(src.size(), src.chs);

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
                double r_t = std::sqrt((double)i * i + (double)j * j);
                if (r_t <= r) {
                    space_weight[max_ofs] = std::exp(r_t * r_t * gauss_space_coeff);
                    space_ofs[max_ofs++] = (int)(i * src.step + j * src.chs);
                }
            }
        }

        double *temp_val = new double[src.chs];

        auto ptr = src.data;
        auto data_len = src.size_ * src.chs;

        for (int i = 0; i < data_len; i += src.chs) {
            double norm = 0;
            int mv = 0;
            for (int k = 0; k < src.chs; ++k) {
                mv += ptr[i + k];
            }

            memset(temp_val, 0, sizeof(double) * src.chs);//清零

            for (int j = 0; j < max_ofs; ++j) {
                double w1 = space_weight[j];

                int cv = 0;
                int c_pos = i + space_ofs[j];
                if ((unsigned)c_pos < (unsigned)data_len) {
                    for (int k = 0; k < src.chs; ++k) {
                        cv += ptr[c_pos + k];
                    }

                    double w2 = color_weight[abs(cv - mv)];
                    double w = w1 * w2;
                    norm += w;
                    for (int k = 0; k < src.chs; ++k) {
                        temp_val[k] += ptr[c_pos + k] * w;
                    }
                }
            }
            for (int k = 0; k < src.chs; ++k) {
                dst.data[i + k] = saturate_cast<_Tp>(temp_val[k] / norm);
            }
        }
    }

    template <typename _Tp> void Laplacian(const _Matrix<_Tp>&src, _Matrix<_Tp>&dst, int ksize)
    {
        assert(ksize > 0 && ksize % 2 == 1);

        z::Matrix8s kernel;
        if (ksize == 1) {
            kernel.create(3, 3);
            kernel = { 0, 1, 0, 1, -4, 1, 0, 1, 0 };
            dst = src.conv(kernel, false);
        }
        else if (ksize == 3) {
            kernel.create(3, 3);
            kernel = { 2, 0, 2, 0, -8, 0, 2, 0, 2 };
            dst = src.conv(kernel, false);
        }
        else {
            assert(1 == 0);
        }
    }


	//////////////////////////////////////形态学滤波//////////////////////////////////////
	template <typename _Tp> void morphOp(int code, _Matrix<_Tp>& src, _Matrix<_Tp>&dst, Size size)
	{
		int area = size.area();
		_Tp ** ker = new _Tp *[src.chs];
		for (int i = 0; i < src.chs; ++i) {
			ker[i] = new _Tp[area];
		}

		if (!src.equalSize(dst))
			dst.create(src.rows, src.cols, src.chs);

		int m = size.width / 2, n = size.height / 2;
//		_Tp * ptr = nullptr;
		_Tp * dstPtr = nullptr;
		int cnt = 0;
		_Tp maxVal = 0;
		_Tp minVal = 0;

		for (auto i = 0; i < src.rows; ++i) {
			for (auto j = 0; j < src.cols; ++j) {

				cnt = 0;
				for (auto ii = 0; ii < size.width; ++ii) {
					for (auto jj = 0; jj < size.height; ++jj) {
                        auto _i = i - m + ii;
                        auto _j = j - n + jj;
						if (_i >= 0 && _i < src.rows && _j >= 0 && _j < src.cols) {
							for (auto k = 0; k < src.chs; ++k) {
								ker[k][cnt] = src.ptr(_i, _j)[k];
							}
							cnt++;
						}
					}
				}
				dstPtr = dst.ptr(i, j);
				switch (code) {
					// 腐蚀， 局部最小值
				case MORP_ERODE:
					for (auto k = 0; k < src.chs; ++k) {
						_min(ker[k], cnt, minVal);
						dstPtr[k] = minVal;
					}
					break;

					// 膨胀，局部最大值
				case MORP_DILATE:
					for (auto k = 0; k < src.chs; ++k) {
						_max(ker[k], cnt, maxVal);
						dstPtr[k] = maxVal;
					}
					break;
				default: ;
				}
				

			} // !for(j)
		} // !for(i)

		for (int i = 0; i < src.chs; ++i) {
			delete[] ker[i];
		}
		delete[] ker;
	}
	
	template <typename _Tp> void erode(_Matrix<_Tp>& src, _Matrix<_Tp>&dst, Size kernel)
	{
		morphOp(MORP_ERODE, src, dst, kernel);
	}

	template <typename _Tp> void dilate(_Matrix<_Tp>& src, _Matrix<_Tp>&dst, Size kernel)
	{
		morphOp(MORP_DILATE, src, dst, kernel);
	}

	template <typename _Tp> void open(_Matrix<_Tp>& src, _Matrix<_Tp>&dst, Size kernel)
	{
		_Matrix<_Tp> _dst;
		morphOp(MORP_ERODE, src, _dst, kernel);
		morphOp(MORP_DILATE, _dst, dst, kernel);
	}

	template <typename _Tp> void morphEx(_Matrix<_Tp>& src, _Matrix<_Tp>&dst, int op, Size kernel)
	{
		_Matrix<_Tp> temp;
		if (dst.equalSize(src))
			dst.create(src.rows, src.cols, src.chs);

		switch (op) {
		case MORP_ERODE:
			erode(src, dst, kernel);
			break;

		case MORP_DILATE:
			dilate(src, dst, kernel);
			break;

		case MORP_OPEN:
			erode(src, temp, kernel);
			dilate(temp, dst, kernel);
			break;

		case MORP_CLOSE:
			dilate(src, temp, kernel);
			erode(temp, dst, kernel);
			break;

		case MORP_BLACKHAT:
			dilate(src, temp, kernel);
			erode(temp, dst, kernel);

			dst -= src;
			break;

		case MORP_TOPHAT:
			erode(src, temp, kernel);
			dilate(temp, dst, kernel);

			dst = src - dst;
			break;

		case MORP_GRADIENT:
			dilate(src, temp, kernel);
			erode(temp, dst, kernel);

			dst = temp - dst;
			break;
		default: ;
		}
	}

	/**
	 * @brief 将多通道矩阵分离称为单通道的矩阵
	 */
	template <typename _Tp> void spilt(_Matrix<_Tp> & src, std::vector<_Matrix<_Tp>> & mv)
	{
		mv = std::vector<_Matrix<_Tp>>(src.chs);

		for (int i = 0; i < src.chs; ++i) {
			mv.at(i).create(src.rows, src.cols, 1);
		}

		for (int i = 0; i < src.rows; ++i) {
			for (int j = 0; j < src.cols; ++j) {
				for (int k = 0; k < src.chs; ++k) {
					mv.at(k).ptr(i, j)[0] = src.ptr(i, j)[k];
				}
			}
		}
	}
	/**
	 * @brief 合并两个1通道的矩阵
	 */
	template <typename _Tp> void merge(_Matrix<_Tp> & src1, _Matrix<_Tp> & src2, _Matrix<_Tp> & dst)
	{
		if (!src1.equalSize(src2))
			_log_("!src1.equalSize(src2)");

		if (dst.rows != src1.rows || dst.cols != src1.cols)
			dst.create(src1.rows, src1.cols, 2);

		for (int i = 0; i < src1.rows; ++i) {
			for (int j = 0; j < src2.cols; ++j) {
				dst.ptr(i, j)[0] = src1.ptr(i, j)[0];
				dst.ptr(i, j)[1] = src2.ptr(i, j)[0];
			}
		}
	}

	/**
	 * @brief 合并通道，顺序按照src中的顺序
	 */
	template <typename _Tp> void merge(std::vector<_Matrix<_Tp>> & src, _Matrix<_Tp> & dst)
	{
		if (src.size() < 1)
			_log_("src.size() < 1");

		int rows = src.at(0).rows;
		int cols = src.at(0).cols;
		int chs = src.size();

		// 检查
		for (int i = 1; i < chs; ++i) {
			if(src.at(i).rows != rows || src.at(i).cols != cols)
				_log_("src.at(i).rows != rows || src.at(i).cols != cols");
		}

		// 是否需要分配内存
		if(dst.rows != rows || dst.cols != cols || dst.chs != chs)
			dst.create(rows, cols, chs);

		// 合并
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				for (int k = 0; k < chs; ++k) {
					dst.ptr(i, j)[k] = src.at(k).ptr(i, j)[0];
				} // !for(k)
			} // !for(j)
		} // !for(i)
	}

	template <typename _Tp> void copyMakeBorder(_Matrix<_Tp> & src, _Matrix<_Tp> & dst, int top, int bottom, int left, int right)
	{
		dst.create(src.rows + top + bottom, src.cols + left + right, src.chs);
		dst.init(0);
		_Tp * srcPtr, *dstPtr;

		for (int i = 0; i < dst.rows; ++i) {
			for (int j = 0; j < dst.cols; ++j) {
				dstPtr = dst.ptr(i, j);
				if (i >= top && j >= left && i < src.rows + top && j < src.cols + left) {
					srcPtr = src.ptr(i - top, j - left);
					for (int k = 0; k < dst.chs; ++k) {
						dstPtr[k] = srcPtr[k];
					}
				}
				else {
					for (int k = 0; k < dst.chs; ++k) {
						dstPtr[k] = 0;
					}
				}
			}
		}
	}


    template <typename _Tp> void threshold(_Matrix<_Tp> &src, _Matrix<_Tp>& dst, double thresh, double maxval, int type)
    {
        assert(src.chs == 1);

        if (!dst.equalSize(src))
            dst.create(src.size(), src.chs);

        auto srcptr = src.datastart;
        auto dstptr = dst.datastart;

        switch (type) {
        case THRESH_BINARY:
            for (int i = 0; srcptr + i < src.dataend; ++i)
                srcptr[i] > _Tp(thresh) ? dstptr[i] = _Tp(maxval) : dstptr[i] = _Tp(0);
            break;

        case THRESH_BINARY_INV:
            for (int i = 0; srcptr + i < src.dataend; ++i)
                srcptr[i] > _Tp(thresh) ? dstptr[i] = _Tp(0) : dstptr[i] = _Tp(maxval);
            break;

        case THRESH_TRUNC:
            for (int i = 0; srcptr + i < src.dataend; ++i)
                srcptr[i] > _Tp(thresh) ? dstptr[i] = _Tp(thresh) : dstptr[i] = _Tp(0);
            break;

        case THRESH_TOZERO:
            for (int i = 0; srcptr + i < src.dataend; ++i)
                srcptr[i] > _Tp(thresh) ? dstptr[i] = srcptr[i] : dstptr[i] = _Tp(0);
            break;

        case THRESH_TOZERO_INV:
            for (int i = 0; srcptr + i < src.dataend; ++i)
                srcptr[i] > _Tp(thresh) ? dstptr[i] = _Tp(0) : dstptr[i] = srcptr[i];
            break;
        }
    }


    template <typename _Tp> void pyrUp(const _Matrix<_Tp>& src, _Matrix<_Tp>& dst)
    {
        Matrix temp;

        Matrix64f ker(5, 5, 1);
        ker = {
            1, 4, 6, 4, 1,
            4, 16, 24, 16, 4,
            6, 24, 36, 24, 6,
            4, 16, 24, 16, 4,
            1, 4, 6, 4, 1
        };

        int dst_rows = src.rows * 2;
        int dst_cols = src.cols * 2;

        temp.create(dst_rows, dst_cols, src.chs);
        temp.zeros();

        for (int i = 0; i < src.rows; ++i) 
            for (int j = 0;j < src.cols; ++j) 
                for (int k = 0; k < src.chs; ++k) 
                    temp.ptr(2 * i, 2 * j)[k] = src.ptr(i, j)[k];


        dst = temp.conv(ker, true);
        for (int i = 0; i < dst.rows; ++i)
            for (int j = 0; j < dst.cols; ++j) 
                for (int k = 0;k < dst.chs; ++k) 
                    dst.ptr(i, j)[k] = saturate_cast<_Tp>(dst.ptr(i, j)[k] * 4);
    }


    template <typename _Tp> void pyrDown(const _Matrix<_Tp>& src, _Matrix<_Tp>& dst)
    {
        Matrix temp = src.clone();

        Matrix64f ker(5, 5, 1);
        ker = {
            1, 4, 6, 4, 1,
            4, 16, 24, 16, 4,
            6, 24, 36, 24, 6,
            4, 16, 24, 16, 4,
            1, 4, 6, 4, 1
        };
        temp = src.conv(ker, true);

        int dst_rows = src.rows / 2;
        int dst_cols = src.cols / 2;

        dst.create(dst_rows, dst_cols, src.chs);
        for (auto i = 0; i < dst_rows; ++i) 
            for (auto j = 0;j < dst_cols; ++j) 
                for (auto k = 0; k < src.chs; ++k) 
                    dst.ptr(i, j)[k] = src.ptr(2 * i, 2 * j)[k];
    }
};

#endif // !_ZIMGPROC_HPP 