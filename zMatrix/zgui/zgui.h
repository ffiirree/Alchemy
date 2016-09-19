#ifndef _ZGUI_H
#define _ZGUI_H


#include "zcore.h"

namespace z {
Matrix8u imread(char *filename);
void imwrite(char *filename, Matrix8u & img, int quality = 95);
extern "C" {
int ReadJpegQuality(const char *filename);
}

}

#endif