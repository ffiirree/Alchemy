#ifndef _ZIMGPROC_HPP
#define _ZIMGPROC_HPP

#include <algorithm>
#include <vector>

namespace z {

	template<class _Tp> inline _Size<_Tp>& _Size<_Tp>::operator = (const _Size& sz)
	{
		width = sz.width;
		height = sz.height;
		return *this; 
	}

	template <class _type> void cvtColor(_Matrix<_type>&src, _Matrix<_type>&dst, int code)
	{
		switch (code) {
		case BGR2GRAY:
		{
			if (!(dst.rows == src.rows && dst.cols == src.cols && dst.chs == 1 && src.chs == 3))
				dst.create(src.rows, src.cols, 1);

			double mediaValue = 0;
			_type * srcPtr = nullptr;

			for (int i = 0; i < src.rows; ++i) {
				for (int j = 0; j < src.cols; ++j) {

					srcPtr = src.ptr(i, j);

					dst.ptr(i, j)[0] = _type(0.3 * srcPtr[0] + 0.59 * srcPtr[1] + 0.11 * srcPtr[2]);
				}
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
	template <class _type> void blur(_Matrix<_type>& src, _Matrix<_type>& dst, Size size)
	{
		boxFilter(src, dst, size, true);
	}
	/**
	 * @berif 方框滤波
	 */
	template <class _type> void boxFilter(const _Matrix<_type>& src, _Matrix<_type>& dst, Size size, bool normalize)
	{
		if (size.width != size.height || size.width % 2 == 0)
			throw runtime_error("size.width != size.height || size.width % 2 == 0");

		if (!src.equalSize(dst))
			dst.create(src.rows, src.cols, src.chs);

		int *tempValue = new int[src.chs];
		int zeros = 0;
		int m = size.width / 2, n = size.height / 2;
		const _type * ptr = nullptr;
		_type * dstPtr = nullptr;
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
						dstPtr[k] = (_type)(tempValue[k] / alpha);
					else
						dstPtr[k] = (_type)tempValue;
				}
				

			} // !for(j)
		} // !for(i)

		delete[] tempValue;
	}

	template <class _type> void GaussianBlur(_Matrix<_type>&src, _Matrix<_type> & dst, Size size, double sigmaX, double sigmaY)
	{
		Matrix kernel = Gassion(size, sigmaX, sigmaY);
		src.conv(kernel, dst, true);
	}

	template <class _type> _Matrix<_type> embossingFilter(_Matrix<_type> src, Size size, float ang)
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
	template <class _type> void medianFilter(_Matrix<_type>&src, _Matrix<_type>& dst, Size size)
	{
		int area = size.area();
		_type ** ker = new _type *[src.chs];
		for (int i = 0; i < src.chs; ++i) {
			ker[i] = new _type[area];
		}

		if (!src.equalSize(dst))
			dst.create(src.rows, src.cols, src.chs);

		int m = size.width / 2, n = size.height / 2;
		_type * ptr = nullptr;
		_type * dstPtr = nullptr;
		int cnt = 0;
		int valindex = 0;
		int valDefault = area / 2;

		for (int i = 0; i < src.rows; ++i) {
			for (int j = 0; j < src.cols; ++j) {

				cnt = 0;
				for (int ii = 0; ii < size.width; ++ii) {
					for (int jj = 0; jj < size.height; ++jj) {

						ptr = src.ptr(i - m + ii, j - n + jj);
						if (ptr) {
							for (int k = 0; k < src.chs; ++k) {
								ker[k][cnt] = ptr[k];
							}
							cnt++;
						}
					}
				}
				dstPtr = dst.ptr(i, j);
				if (cnt != area)
					valindex = cnt / 2;
				else
					valindex = valDefault;
				for (int k = 0; k < src.chs; ++k) {
					sort(ker[k], ker[k] + cnt);  // 占95%以上的时间
					dstPtr[k] = ker[k][valindex];
				}

			} // !for(j)
		} // !for(i)

		for (int i = 0; i < src.chs; ++i) {
			delete[] ker[i];
		}
		delete[] ker;
	}


	//////////////////////////////////////形态学滤波//////////////////////////////////////
	template <class _type> void morphOp(int code, _Matrix<_type>& src, _Matrix<_type>&dst, Size size)
	{
		int area = size.area();
		_type ** ker = new _type *[src.chs];
		for (int i = 0; i < src.chs; ++i) {
			ker[i] = new _type[area];
		}

		if (!src.equalSize(dst))
			dst.create(src.rows, src.cols, src.chs);

		int m = size.width / 2, n = size.height / 2;
		_type * ptr = nullptr;
		_type * dstPtr = nullptr;
		int cnt = 0;
		_type maxVal = 0;
		_type minVal = 0;

		for (int i = 0; i < src.rows; ++i) {
			for (int j = 0; j < src.cols; ++j) {

				cnt = 0;
				for (int ii = 0; ii < size.width; ++ii) {
					for (int jj = 0; jj < size.height; ++jj) {

						ptr = src.ptr(i - m + ii, j - n + jj);
						if (ptr) {
							for (int k = 0; k < src.chs; ++k) {
								ker[k][cnt] = ptr[k];
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
						min(ker[k], cnt, minVal);
						dstPtr[k] = minVal;
					}
					break;

					// 膨胀，局部最大值
				case MORP_DILATE:
					for (int k = 0; k < src.chs; ++k) {
						max(ker[k], cnt, maxVal);
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
	
	template <class _type> void erode(_Matrix<_type>& src, _Matrix<_type>&dst, Size kernel)
	{
		morphOp(MORP_ERODE, src, dst, kernel);
	}

	template <class _type> void dilate(_Matrix<_type>& src, _Matrix<_type>&dst, Size kernel)
	{
		morphOp(MORP_DILATE, src, dst, kernel);
	}

	template <class _type> void open(_Matrix<_type>& src, _Matrix<_type>&dst, Size kernel)
	{
		_Matrix<_type> _dst;
		morphOp(MORP_ERODE, src, _dst, kernel);
		morphOp(MORP_DILATE, _dst, dst, kernel);
	}

	template <class _type> void morphEx(_Matrix<_type>& src, _Matrix<_type>&dst, int op, Size kernel)
	{
		_Matrix<_type> temp;
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
};
#endif // !_ZIMGPROC_HPP 