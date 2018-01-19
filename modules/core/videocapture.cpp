#include "videocapture.h"

namespace alchemy
{
VideoCapture::VideoCapture(int32_t index)
{
    opened_ = open(index);
}

bool VideoCapture::open(int32_t index)
{
    index_ = index;

    avcodec_register_all();
    avdevice_register_all();

    format_context_ = avformat_alloc_context();

#if defined _WIN32
    device_name_ = std::to_string(index);
	if ((input_format_ = av_find_input_format("vfwcap")) == nullptr) return false;
#elif  __unix__
    device_name_ = "/dev/video" + std::to_string(index);
    if ((input_format_ = av_find_input_format("v4l2")) == nullptr) return false;
#endif

    if (avformat_open_input(&format_context_, device_name_.c_str(), input_format_, nullptr) < 0) return false;
    if(avformat_find_stream_info(format_context_, nullptr) < 0) return false;

    AVCodec *codec;
    if((video_stream_index_ = av_find_best_stream(format_context_, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0)) < 0)
        return false;

    codec_context_ = format_context_->streams[video_stream_index_]->codec;
    if(avcodec_open2(codec_context_, codec, nullptr) < 0) return false;

    avpicture_alloc(&picture_, AV_PIX_FMT_BGR24, codec_context_->width, codec_context_->height);
    frame_ = av_frame_alloc();

    return true;
}

VideoCapture::~VideoCapture()
{
    av_packet_unref(&packet_);
    avcodec_close(codec_context_);
    av_free(frame_);
    avpicture_free(&picture_);
    avformat_close_input(&format_context_);
}

VideoCapture &VideoCapture::operator>>(alchemy::Matrix &image)
{
    read(image);
    return *this;
}

void VideoCapture::read(Matrix &image)
{
    int unfinished = 0;
    struct SwsContext * img_convert_ctx = nullptr;

    if(av_read_frame(format_context_, &packet_) >= 0){
        if(packet_.stream_index == video_stream_index_) {
            avcodec_decode_video2(codec_context_, frame_, &unfinished, &packet_);

            if(unfinished) {
                img_convert_ctx = sws_getCachedContext(
                        img_convert_ctx,
                        codec_context_->width, codec_context_->height, codec_context_->pix_fmt,
                        codec_context_->width, codec_context_->height, AV_PIX_FMT_BGR24,
                        SWS_BICUBIC, nullptr, nullptr, nullptr
                );

                sws_scale(
                        img_convert_ctx,
                        reinterpret_cast<const uint8_t *const *>(frame_->data), frame_->linesize,
                        0, codec_context_->height,
                        picture_.data, picture_.linesize
                );

                image.create(frame_->height, frame_->width, 3);
                memcpy(image.data, picture_.data[0], image.total() * image.channels());
            }
        }
    }

    av_free_packet(&packet_);
    sws_freeContext(img_convert_ctx);
}
}