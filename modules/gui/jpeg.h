#ifndef ALCHEMY_GUI_JPEG_H
#define ALCHEMY_GUI_JPEG_H

#include "core/matrix.h"

namespace alchemy {
int read_JPEG_file(const char * filename, alchemy::Matrix & img);
void write_JPEG_file(const char * filename, alchemy::Matrix & img, int quality);
}

#endif //! ALCHEMY_GUI_JPEG_H
