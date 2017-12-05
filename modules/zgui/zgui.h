#ifndef _ZGUI_ZGUI_H
#define _ZGUI_ZGUI_H


#include "zimgproc/zimgproc.h"

namespace z {

Matrix imread(const std::string& filename);
void imwrite(const std::string& filename, Matrix8u & img, int quality = 95);

void imshow(const std::string & name, const Matrix8u & mat);
void namedWindow(const std::string & name, int flags = 1);
int waitKey(int delay = 0);


// line
void lineDDA(Matrix8u & img, Point pt1, Point pt2, const Scalar8u& color, int thickness = 1);
void line(Matrix8u & img, Point pt1, Point pt2, const Scalar8u& color, int thickness = 1);
}

#endif //!_ZGUI_ZGUI_H