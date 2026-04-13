#ifndef RKNN_YOLOV5_DEMO_MEDIA_FFMPEG_DEMUXER_H_
#define RKNN_YOLOV5_DEMO_MEDIA_FFMPEG_DEMUXER_H_

#include <atomic>
#include <stdint.h>

#include <memory>
#include <string>

#include "media/ffmpeg_headers.h"

// 文件说明：
// 定义 FFmpeg demux 封装，用来打开输入源并持续读取视频压缩包。

namespace rknn_demo {

// 描述探测到的输入视频流参数。
struct MediaStreamInfo {
    // 视频编码类型。
    AVCodecID codec_id;

    // 视频宽度。
    int width;

    // 视频高度。
    int height;

    // 输入流时间基。
    AVRational time_base;

    // 对视频流编码参数做一份持久化拷贝，供解码器初始化时直接复用。
    std::shared_ptr<AVCodecParameters> codec_parameters;

    // 构造一份空的流信息，避免未初始化字段被误用。
    MediaStreamInfo()
        : codec_id(AV_CODEC_ID_NONE),
          width(0),
          height(0),
          time_base(av_make_q(0, 1)) {}
};

// `FfmpegDemuxer` 负责打开输入流并从中读取视频压缩包。
class FfmpegDemuxer {
public:
    // 构造一个空 demuxer，初始时未绑定任何输入流。
    FfmpegDemuxer();

    // 析构时自动关闭输入流并释放相关资源。
    ~FfmpegDemuxer();

    // 打开输入源，并把探测到的视频流信息写入 `stream_info`。
    bool Open(const std::string &input, MediaStreamInfo *stream_info);

    // 配置一个可选的中断标记。
    // 当该标记变为 true 时，阻塞中的 FFmpeg IO 会尽快退出。
    void SetInterruptFlag(const std::atomic<bool> *interrupt_flag);

    // 读取下一份视频压缩包到 `packet`。
    bool ReadPacket(AVPacket *packet);

    // 关闭当前输入流并释放相关资源。
    void Close();

private:
    // FFmpeg 输入格式上下文。
    AVFormatContext *format_context_;

    // 当前选中的视频流索引。
    int video_stream_index_;

    // 供 FFmpeg interrupt callback 查询的外部停止标记。
    const std::atomic<bool> *interrupt_flag_;
};

}  // namespace rknn_demo

#endif  // RKNN_YOLOV5_DEMO_MEDIA_FFMPEG_DEMUXER_H_
