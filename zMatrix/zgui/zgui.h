#ifndef _ZGUI_H
#define _ZGUI_H


#include "zcore.h"
#include <string>
#include "zimgproc\zimgproc.h"

enum
{
	//These 3 flags are used by cvSet/GetWindowProperty
	WND_PROP_FULLSCREEN = 0, //to change/get window's fullscreen property
	WND_PROP_AUTOSIZE = 1, //to change/get window's autosize property
	WND_PROP_ASPECTRATIO = 2, //to change/get window's aspectratio property
	WND_PROP_OPENGL = 3, //to change/get window's opengl support

	//These 2 flags are used by cvNamedWindow and cvSet/GetWindowProperty
	WINDOW_NORMAL = 0x00000000, //the user can resize the window (no constraint)  / also use to switch a fullscreen window to a normal size
	WINDOW_AUTOSIZE = 0x00000001, //the user cannot resize the window, the size is constrainted by the image displayed
	WINDOW_OPENGL = 0x00001000, //window with opengl support

	//These 3 flags are used by cvNamedWindow and cvSet/GetWindowProperty
	WINDOW_FULLSCREEN = 1,//change the window to fullscreen
	WINDOW_FREERATIO = 0x00000100,//the image expends as much as it can (no ratio constraint)
	WINDOW_KEEPRATIO = 0x00000000//the ration image is respected.
};

namespace z {

Matrix8u imread(char *filename);
void imwrite(char *filename, Matrix8u & img, int quality = 95);

void imshow(const std::string & name, Matrix8u & mat);
void namedWindow(const std::string & name, int flags = 1);
int waitKey(int delay = 0);


// 图形绘制
void line(Matrix8u & img, Point pt1, Point pt2, const Scalar& color, int thickness = 1, int lineType = 8, int shift = 0);
void ellipse(Matrix8u& img, Point center, Size axes, double angle, const Scalar& color, int thickness = 1, int lineType = 8, int shift = 0);
void rectangle(Matrix8u& img, Rect rec, const Scalar& color, int thickness = 1, int lineType = 8, int shift = 0);
void circle(Matrix8u& img, Point center, int radius, const Scalar& color, int thickness = 1, int lineType = 8, int shift = 0);
}

#endif