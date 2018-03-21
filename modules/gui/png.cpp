#include "png.h"
#include <glog/logging.h>

extern "C" {
#include <png.h>
#include <zlib.h>
}

namespace alchemy {
int read_PNG_file(const char * filename, Matrix & img)
{
    FILE * fp = fopen(filename, "rb");
    if(!fp) {
        fprintf(stderr, "can't open %s\n", filename);
        return -1;
    }

    uint8_t sig[8];
    if(!fread(sig, 1, 8, fp)) {
        fclose(fp);
        fprintf(stderr, "can't read sig.");
        return -1;
    }

    bool is_png = !png_sig_cmp(sig, 0, 8);
    if(!is_png) {
        fclose(fp);
        fprintf(stderr, "not png.");
        return -1;
    }

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if(!png_ptr) {
        fclose(fp);
        return -1;
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if(!info_ptr) {
        fclose(fp);
        png_destroy_read_struct(&png_ptr, nullptr, nullptr);
        return -1;
    }

    if(setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
        fclose(fp);
        return -1;
    }

    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, 8);

    png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_BGR, nullptr);
    img.create(static_cast<int>(info_ptr->height), static_cast<int>(info_ptr->width), info_ptr->channels);
    auto row_pointers = png_get_rows(png_ptr, info_ptr);

    for(uint32_t i = 0; i < info_ptr->height; ++i) {
        memcpy(img.data + img.step * i, row_pointers[i], static_cast<size_t>(img.step));
    }
    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    fclose(fp);

    return 0;
}
void write_PNG_file(const char * filename, Matrix & img)
{
    LOG(FATAL) << "Not implemented!";
}
}