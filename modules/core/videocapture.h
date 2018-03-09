#ifndef ALCHEMY_CORE_VIDEOCAPTURE_H
#define ALCHEMY_CORE_VIDEOCAPTURE_H

#include <string>

extern "C" {
#include <libswscale/swscale.h>
#include <libavdevice/avdevice.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
};

#include "matrix.h"

namespace alchemy {

class VideoCapture {
public:
    explicit VideoCapture(int32_t index);
    explicit VideoCapture(const std::string& file);
    VideoCapture(const VideoCapture&) = delete;
    VideoCapture& operator=(const VideoCapture&) = delete;
    ~VideoCapture();

    bool isOpened() const { return opened_; }

    bool open(int32_t index);
    bool open(const std::string& file);

    void read(Matrix& image);

    VideoCapture& operator >> (alchemy::Matrix& image);

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

}


#endif //! ALCHEMY_CORE_VIDEOCAPTURE_H
