#ifndef _ZGUI_WINDOWS_WIN32_H
#define _ZGUI_WINDOWS_WIN32_H

#define Z_WINDOW_MAGIC_VAL     0x00420042
#define Z_TRACKBAR_MAGIC_VAL   0x00420043

typedef void (* zMouseCallback)(int event, int x, int y, int flags, void* param);
typedef void (* zTrackbarCallback)(int pos);
typedef void (* zTrackbarCallback2)(int pos, void* userdata);
typedef void (* zMouseCallback)(int event, int x, int y, int flags, void* param);

struct zRect
{
    zRect() = default;
    zRect(int _x, int _y, int _width, int _height)
        :x(_x), y(_y), width(_width), height(_height) 
    {}

public:
	int x = 0;
	int y = 0;
	int width = 0;
	int height = 0;
};


int zNamedWindow(const char* name, int flags = 1);
int zWaitKey(int delay);
void zShowImage(const char* name, void* arr);
int GUIInitSystem(int argc, char** argv);



#endif // !_ZGUI_WINDOWS_WIN32_H