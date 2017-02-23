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


#ifdef __cplusplus
namespace z {

	template<class _Tp> inline _Size<_Tp>& _Size<_Tp>::operator = (const _Size& sz)
	{
		width = sz.width;
		height = sz.height;
		return *this; 
	}

	template <class _Tp> void cvtColor(const _Matrix<_Tp>&src, _Matrix<_Tp>&dst, int code)
	{
		switch (code) {
		case BGR2GRAY:
		{
			if (!(dst.rows == src.rows && dst.cols == src.cols && dst.chs == 1 && src.chs == 3))
				dst.create(src.rows, src.cols, 1);

			const _Tp * srcPtr = nullptr;

			for (int i = 0; i < src.rows; ++i) {
				for (int j = 0; j < src.cols; ++j) {

					srcPtr = src.ptr(i, j);

					dst.ptr(i, j)[0] = _Tp(0.3 * srcPtr[0] + 0.59 * srcPtr[1] + 0.11 * srcPtr[2]);
				}
			}
		}
			break;

		case BGR2RGB:
			if (!dst.equalSize(src)) {
				dst.create(src.rows, src.cols, src.chs);
			}

			for (int i = 0; i < src.rows; ++i) {
				for (int j = 0; j < src.cols; ++j) {
					dst.ptr(i, j)[2] = src.ptr(i, j)[0];
					dst.ptr(i, j)[1] = src.ptr(i, j)[1];
					dst.ptr(i, j)[0] = src.ptr(i, j)[2];
				}
			}
			break;

		default:
			break;
		}
	}

	/**
	 * @berif 均值滤波
	 */
	template <class _Tp> void blur(_Matrix<_Tp>& src, _Matrix<_Tp>& dst, Size size)
	{
		boxFilter(src, dst, size, true);
	}

	/**
	 * @berif 方框滤波
	 * @param[in] normalize，是否归一化，卷积核各项和不为1时除以和。
	 */
	template <class _Tp> void boxFilter(const _Matrix<_Tp>& src, _Matrix<_Tp>& dst, Size size, bool normalize)
	{
		if (size.width != size.height || size.width % 2 == 0)
			_log_("size.width != size.height || size.width % 2 == 0");

		if (!src.equalSize(dst))
			dst.create(src.rows, src.cols, src.chs);

		int *tempValue = new int[src.chs];
		int zeros = 0;
		int m = size.width / 2, n = size.height / 2;
		const _Tp * ptr = nullptr;
		_Tp * dstPtr = nullptr;
		int alpha = 0;

		for (int i = 0; i < dst.rows; ++i) {
			for (int j = 0; j < dst.cols; ++j) {

				memset(tempValue, 0, src.chs * sizeof(int));
				zeros = 0;

				for (int ii = 0; ii < size.width; ++ii) {
					for (int jj = 0; jj < size.height; ++jj) {

						// 获取一个像素的地址
						ptr = src.ptr(i - m + ii, j - n + jj);

						if (ptr) {
							for (int k = 0; k < src.chs; ++k) {
								tempValue[k] += ptr[k];
							}
						}
						else {
							zeros++;
						}
					} // !for(jj)
				} // !for(ii)

				alpha= size.area() - zeros;

				dstPtr = dst.ptr(i,j);

				for (int k = 0; k < src.chs; ++k) {
					if (normalize)
						dstPtr[k] = (_Tp)(tempValue[k] / alpha);
					else
						dstPtr[k] = (_Tp)tempValue;
				}
				

			} // !for(j)
		} // !for(i)

		delete[] tempValue;
	}

	/**
	 * @berif 高斯滤波
	 * @param[in] normalize，是否归一化，卷积核各项和不为1时除以和。
	 */
	template <class _Tp> void GaussianBlur(_Matrix<_Tp>&src, _Matrix<_Tp> & dst, Size size, double sigmaX, double sigmaY)
	{
        Matrix64f kernel = Gassion(size, sigmaX, sigmaY);
		src.conv(kernel, dst, true);
	}

	template <class _Tp> _Matrix<_Tp> embossingFilter(_Matrix<_Tp> src, Size size, float ang)
	{
		Matrix kernel(size);

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
		return src.conv(kernel);
	}
	

	/**
	 * @berif 中值滤波
	 */
	template <class _Tp> void medianFilter(_Matrix<_Tp>&src, _Matrix<_Tp>& dst, Size size)
	{
		int area = size.area();
		_Tp ** ker = new _Tp *[src.chs];
		for (int i = 0; i < src.chs; ++i) {
			ker[i] = new _Tp[area];
		}

		if (!src.equalSize(dst))
			dst.create(src.rows, src.cols, src.chs);

		int m = size.width / 2, n = size.height / 2;
		_Tp * ptr = nullptr;
		_Tp * dstPtr = nullptr;
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
				if (cnt != area)
					valindex = cnt / 2;
				else
					valindex = valDefault;
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


	//////////////////////////////////////形态学滤波//////////////////////////////////////
	template <class _Tp> void morphOp(int code, _Matrix<_Tp>& src, _Matrix<_Tp>&dst, Size size)
	{
		int area = size.area();
		_Tp ** ker = new _Tp *[src.chs];
		for (int i = 0; i < src.chs; ++i) {
			ker[i] = new _Tp[area];
		}

		if (!src.equalSize(dst))
			dst.create(src.rows, src.cols, src.chs);

		int m = size.width / 2, n = size.height / 2;
		_Tp * ptr = nullptr;
		_Tp * dstPtr = nullptr;
		int cnt = 0;
		_Tp maxVal = 0;
		_Tp minVal = 0;

		for (int i = 0; i < src.rows; ++i) {
			for (int j = 0; j < src.cols; ++j) {

				cnt = 0;
				for (int ii = 0; ii < size.width; ++ii) {
					for (int jj = 0; jj < size.height; ++jj) {
                        auto _i = i - m + ii;
                        auto _j = j - n + jj;
						if (_i >= 0 && _i < src.rows && _j >= 0 && _j < src.cols) {
							for (int k = 0; k < src.chs; ++k) {
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
					for (int k = 0; k < src.chs; ++k) {
						_min(ker[k], cnt, minVal);
						dstPtr[k] = minVal;
					}
					break;

					// 膨胀，局部最大值
				case MORP_DILATE:
					for (int k = 0; k < src.chs; ++k) {
						_max(ker[k], cnt, maxVal);
						dstPtr[k] = maxVal;
					}
					break;
				}
				

			} // !for(j)
		} // !for(i)

		for (int i = 0; i < src.chs; ++i) {
			delete[] ker[i];
		}
		delete[] ker;
	}
	
	template <class _Tp> void erode(_Matrix<_Tp>& src, _Matrix<_Tp>&dst, Size kernel)
	{
		morphOp(MORP_ERODE, src, dst, kernel);
	}

	template <class _Tp> void dilate(_Matrix<_Tp>& src, _Matrix<_Tp>&dst, Size kernel)
	{
		morphOp(MORP_DILATE, src, dst, kernel);
	}

	template <class _Tp> void open(_Matrix<_Tp>& src, _Matrix<_Tp>&dst, Size kernel)
	{
		_Matrix<_Tp> _dst;
		morphOp(MORP_ERODE, src, _dst, kernel);
		morphOp(MORP_DILATE, _dst, dst, kernel);
	}

	template <class _Tp> void morphEx(_Matrix<_Tp>& src, _Matrix<_Tp>&dst, int op, Size kernel)
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
		}
	}

	/**
	 * @berif 将多通道矩阵分离称为单通道的矩阵
	 */
	template <class _Tp> void spilt(_Matrix<_Tp> & src, std::vector<_Matrix<_Tp>> & mv)
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
	 * @berif 合并两个1通道的矩阵
	 */
	template <class _Tp> void merge(_Matrix<_Tp> & src1, _Matrix<_Tp> & src2, _Matrix<_Tp> & dst)
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
	 * @berif 合并通道，顺序按照src中的顺序
	 */
	template <class _Tp> void merge(std::vector<_Matrix<_Tp>> & src, _Matrix<_Tp> & dst)
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

	template <class _Tp> void copyMakeBorder(_Matrix<_Tp> & src, _Matrix<_Tp> & dst, int top, int bottom, int left, int right)
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
};
#endif

#endif // !_ZIMGPROC_HPP 