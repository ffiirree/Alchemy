#ifndef ALCHEMY_GUI_PNG_H
#define ALCHEMY_GUI_PNG_H

#include "core/matrix.h"

namespace alchemy {
int read_PNG_file(const char * filename, Matrix& img);
void write_PNG_file(const char * filename, Matrix & img);
}

#endif //! ALCHEMY_GUI_PNG_H
