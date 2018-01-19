#ifndef ALCHEMY_GUI_WINDOWS_WIN32_H
#define ALCHEMY_GUI_WINDOWS_WIN32_H

#include "core/types.h"

#define Z_WINDOW_MAGIC_VAL     0x00420042
#define Z_TRACKBAR_MAGIC_VAL   0x00420043

typedef void (* zMouseCallback)(int event, int x, int y, int flags, void* param);
typedef void (* zTrackbarCallback)(int pos);
typedef void (* zTrackbarCallback2)(int pos, void* userdata);
typedef void (* zMouseCallback)(int event, int x, int y, int flags, void* param);

int zNamedWindow(const char* name, int flags = 1);
int zWaitKey(int delay);
void zShowImage(const char* name, const void* arr);
int GUIInitSystem(int argc, char** argv);

#endif //! ALCHEMY_GUI_WINDOWS_WIN32_H