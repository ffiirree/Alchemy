#include "gui.h"
#include "jpeg.h"

#ifdef __unix__
#ifdef USE_GTK2
#include "window_gtk.h"
#endif //! USE_GTK2
#elif defined(_WIN32) || defined(WIN32)
#include "window_win32.h"
#endif //! __unix__



namespace alchemy{

Matrix imread(const std::string& filename)
{
	Matrix temp;
	read_JPEG_file(filename.c_str(), temp);
	return temp;
}

void imwrite(const std::string& filename, Matrix8u & img, int quality)
{
	Matrix8u rgbimg;
	cvtColor(img, rgbimg, BGR2RGB);

	write_JPEG_file(filename.c_str(), rgbimg, quality);
}

void namedWindow(const std::string & name, int flags)
{
	zNamedWindow(name.c_str(), flags);
}

void imshow(const std::string & name, const Matrix8u & mat)
{
	assert(mat.cols != 0 && mat.rows != 0 && mat.data != nullptr);
	assert(mat.channels() == 1 || mat.channels() == 3);
	assert(!name.empty());

	zShowImage(name.c_str(), &mat);
}

int waitKey(int delay)
{
	return zWaitKey(delay);
}


void lineDDA(Matrix8u & img, Point pt1, Point pt2, const Scalar8u& color, int thickness)
{
	__unused_parameter__(thickness);

	auto x = static_cast<float>(pt1.x), y = static_cast<float>(pt1.y);
	int dx = pt2.x - pt1.x;
	int dy = pt2.y - pt1.y;
	float steps = 0;

	if (std::abs(dx) > std::abs(dy))
		steps = std::abs(dx);
	else
		steps = std::abs(dy);

	float xi = dx / steps;
	float yi = dy / steps;

	for (int i = 0; i < steps; ++i) {
		for (int k = 0; k < img.channels(); ++k) {
			img.at(static_cast<int>(x), static_cast<int>(y), img.channels() - k - 1) = color.v[k];
		}
		x += xi;
		y += yi;
	}
}

void line(Matrix8u & img, Point pt1, Point pt2, const Scalar8u& color, int thickness)
{
	lineDDA(img, pt1, pt2, color, thickness);
}

}





