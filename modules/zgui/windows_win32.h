#ifndef _WINDOWS_WIN32_H
#define _WINDOWS_WIN32_H

#include <Windows.h>
#include "zcore/zmatrix.h"

#define CV_WINDOW_MAGIC_VAL     0x00420042
#define CV_TRACKBAR_MAGIC_VAL   0x00420043

typedef void (CV_CDECL *zMouseCallback)(int event, int x, int y, int flags, void* param);

typedef struct zWindow
{
	int signature;
	HWND hwnd;
	char *name;
	zWindow *prev;
	zWindow *next;

	HDC hdc;
	HGDIOBJ image;
	int last_key;
	int status;
	int flags;

	zMouseCallback on_mouse;

	int width;
	int height;
}zWindow;

typedef struct zRect
{
	int x;
	int y;
	int width;
	int height;
}zRect;

int zNamedWindow(const char *name, int flags = 1);

int zWaitKey(int delay);
void zShowImage(const char *name, void * arr);


int initSystem(int argc, char** argv);



#endif