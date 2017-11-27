//
// Created by ffiirree on 17-11-22.
//

#ifndef _ZCORE_VIDEOCAPTURE_H
#define _ZCORE_VIDEOCAPTURE_H

#include <string>
#include "matrix.h"

#ifdef USE_FFMPEG

extern "C" {
#include "libswscale/swscale.h"
#include "libavdevice/avdevice.h"
#include "libavformat/avformat.h"
#include "libavcodec/avcodec.h"
};

namespace z {

class VideoCapture {
public:
    explicit VideoCapture(int32_t index);
    VideoCapture(const VideoCapture&) = delete;
    VideoCapture& operator=(const VideoCapture&) = delete;
    ~VideoCapture();

    bool isOpened() const { return opened_; }

    bool open(int32_t index);

    void read(Matrix& image);

    VideoCapture& operator >> (z::Matrix& image);

private:
    AVFormatContext * format_context_ = nullptr;
    AVInputFormat * input_format_ = nullptr;
    AVCodecContext * codec_context_ = nullptr;
    AVPicture picture_{};
    AVFrame * frame_ = nullptr;
    AVPacket packet_{};

    int32_t index_ = 0;
    std::string device_name_;
    int32_t video_stream_index_ = 0;

    bool opened_ = false;
};

#endif //! USE_FFMPEG
}


#endif //! _ZCORE_VIDEOCAPTURE_H
