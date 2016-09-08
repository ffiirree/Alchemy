#ifndef _ZIMGPROC_HPP
#define _ZIMGPROC_HPP

#include <algorithm>

namespace z {

	template<class _Tp> inline _Size<_Tp>& _Size<_Tp>::operator = (const _Size& sz)
	{
		width = sz.width;
		height = sz.height;
		return *this; 
	}

	/**
	 * @berif 均值滤波
	 */
	template <class _type> _Matrix<_type> blur(_Matrix<_type> src, Size size)
	{
		Matrix kernel(size);
		kernel.init(1.0);

		return src.conv(kernel, size.area());
	}
	
	/**
	 * @berif 方框滤波
	 */
	template <class _type> _Matrix<_type> boxFilter(_Matrix<_type> src, Size size, bool normalize)
	{
		Matrix kernel(size);
		kernel.init(1.0);

		if(normalize)
			return src.conv(kernel, size.area());
		else
			return src.conv(kernel);
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
	template <class _type> _Matrix<_type> medianFilter(_Matrix<_type> src, Size size)
	{
		Matrix kernel(size);
		_Matrix<_type> temp = src.clone();

		for (int i = 0; i < src.rows; ++i) {
			for (int j = 0; j < src.cols; ++j) {


				for (int k = 0; k < kernel.rows; ++k) {
					for (int l = 0; l < kernel.cols; ++l) {
						kernel[k][l] = src.at(i - kernel.rows/ 2 + k, j - kernel.cols/2 + l);
					}
				}
				sort(kernel.data, kernel.data + kernel.size()-1);
				temp[i][j] = kernel[kernel.rows / 2 + 1][kernel.cols / 2 + 1];

			}
		}

		return temp;

	}
}

#endif // !_ZIMGPROC_HPP 