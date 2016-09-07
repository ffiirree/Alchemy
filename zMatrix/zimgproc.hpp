#ifndef _ZIMGPROC_HPP
#define _ZIMGPROC_HPP

namespace z {

	template<class _Tp> inline _Size<_Tp>& _Size<_Tp>::operator = (const _Size& sz)
	{
		width = sz.width;
		height = sz.height;
		return *this; 
	}

	template <class _type> _Matrix<_type> blur(_Matrix<_type> src, Size size)
	{
		Matrix kernel(size.width, size.height);
		kernel.init(1.0);

		return src.conv(kernel, size.area());
	}
	
}

#endif // !_ZIMGPROC_HPP 