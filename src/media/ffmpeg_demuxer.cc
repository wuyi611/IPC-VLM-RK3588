#include "media/ffmpeg_demuxer.h"

#include <stdio.h>
#include <string.h>

#include "media/ffmpeg_headers.h"

// 文件说明：
// 实现 FFmpeg demux 逻辑，并把 FFmpeg 的错误信息转换成人类可读文本。

namespace rknn_demo {

namespace {

// 把 FFmpeg 错误码转换成易读文本并打印到日志。
void PrintFfmpegError(const char *prefix, int err) {
    if (err == AVERROR_EXIT) {
        printf("%s: interrupted by stop request\n", prefix);
        return;
    }

    char errbuf[256] = {0};  // 接收 `av_strerror` 输出的错误描述。
    av_strerror(err, errbuf, sizeof(errbuf));
    printf("%s: %s\n", prefix, errbuf);
}

// 供 FFmpeg 阻塞式 IO 轮询的中断回调。
int InterruptCallback(void *opaque) {
    const std::atomic<bool> *interrupt_flag =
        static_cast<const std::atomic<bool> *>(opaque);  // 外部传入的停止标记。
    if (interrupt_flag == NULL) {
        return 0;
    }
    return interrupt_flag->load(std::memory_order_relaxed) ? 1 : 0;
}

}  // namespace

// 构造一个尚未打开输入流的 demuxer。
FfmpegDemuxer::FfmpegDemuxer()
    : format_context_(NULL),
      video_stream_index_(-1),
      interrupt_flag_(NULL) {}

// 析构时兜底释放 FFmpeg 输入上下文。
FfmpegDemuxer::~FfmpegDemuxer() {
    Close();
}

// 打开输入源并探测其中最合适的视频流信息。
bool FfmpegDemuxer::Open(const std::string &input, MediaStreamInfo *stream_info) {
    if (stream_info == NULL) {
        return false;
    }

    avformat_network_init();
    Close();

    format_context_ = avformat_alloc_context();
    if (format_context_ == NULL) {
        printf("avformat_alloc_context failed\n");
        return false;
    }
    format_context_->interrupt_callback.callback = &InterruptCallback;
    format_context_->interrupt_callback.opaque =
        const_cast<std::atomic<bool> *>(interrupt_flag_);

    // 打开 RTSP / 网络流时使用的低延迟参数字典。
    AVDictionary *options = NULL;
    av_dict_set(&options, "rtsp_transport", "tcp", 0);
    av_dict_set(&options, "fflags", "nobuffer", 0);
    av_dict_set(&options, "flags", "low_delay", 0);
    av_dict_set(&options, "max_delay", "0", 0);
    av_dict_set(&options, "stimeout", "3000000", 0);
    av_dict_set(&options, "rw_timeout", "3000000", 0);

    // 记录 FFmpeg API 返回码，便于统一错误处理。
    int ret = avformat_open_input(&format_context_, input.c_str(), NULL, &options);
    av_dict_free(&options);
    if (ret < 0) {
        PrintFfmpegError("avformat_open_input failed", ret);
        Close();
        return false;
    }

    // 读取并解析流信息。
    ret = avformat_find_stream_info(format_context_, NULL);
    if (ret < 0) {
        PrintFfmpegError("avformat_find_stream_info failed", ret);
        Close();
        return false;
    }

    // 选择最适合的目标视频流索引。
    video_stream_index_ =
        av_find_best_stream(format_context_, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (video_stream_index_ < 0) {
        PrintFfmpegError("av_find_best_stream failed", video_stream_index_);
        Close();
        return false;
    }

    AVStream *stream = format_context_->streams[video_stream_index_];  // 当前选中的视频流对象。
    AVCodecParameters *codecpar = stream->codecpar;  // 当前视频流的编码参数。

    stream_info->codec_id = codecpar->codec_id;
    if (stream_info->codec_id == AV_CODEC_ID_NONE) {
        printf("unsupported codec for ffmpeg rkmpp, codec_id=%d\n", codecpar->codec_id);
        Close();
        return false;
    }

    stream_info->width = codecpar->width;
    stream_info->height = codecpar->height;
    stream_info->time_base = stream->time_base;

    // 为解码器准备一份可长期持有的编码参数副本。
    AVCodecParameters *codec_parameters = avcodec_parameters_alloc();
    if (codec_parameters == NULL) {
        printf("avcodec_parameters_alloc failed\n");
        Close();
        return false;
    }

    ret = avcodec_parameters_copy(codec_parameters, codecpar);
    if (ret < 0) {
        avcodec_parameters_free(&codec_parameters);
        PrintFfmpegError("avcodec_parameters_copy failed", ret);
        Close();
        return false;
    }

    stream_info->codec_parameters.reset(
        codec_parameters,
        [](AVCodecParameters *parameters) {
            avcodec_parameters_free(&parameters);
        });

    printf("demuxer: opened input=%s codec=%s size=%dx%d time_base=%d/%d\n",
           input.c_str(),
           avcodec_get_name(codecpar->codec_id),
           stream_info->width,
           stream_info->height,
           stream_info->time_base.num,
           stream_info->time_base.den);

    return true;
}

// 更新 FFmpeg 中断回调要查询的停止标记。
void FfmpegDemuxer::SetInterruptFlag(const std::atomic<bool> *interrupt_flag) {
    interrupt_flag_ = interrupt_flag;
    if (format_context_ != NULL) {
        format_context_->interrupt_callback.callback = &InterruptCallback;
        format_context_->interrupt_callback.opaque =
            const_cast<std::atomic<bool> *>(interrupt_flag_);
    }
}

// 持续读取输入流，直到拿到一份视频包或者遇到流结束/错误。
bool FfmpegDemuxer::ReadPacket(AVPacket *packet) {
    if (packet == NULL || format_context_ == NULL || video_stream_index_ < 0) {
        return false;
    }

    av_packet_unref(packet);

    while (true) {
        int ret = av_read_frame(format_context_, packet);  // 读取下一份压缩包。
        if (ret < 0) {
            if (ret != AVERROR_EOF) {
                PrintFfmpegError("av_read_frame failed", ret);
            } else {
                printf("demuxer: end of input stream\n");
            }
            return false;
        }

        // 只把视频流的数据包交给调用方，其余流直接丢弃。
        if (packet->stream_index == video_stream_index_) {
            return true;
        }

        av_packet_unref(packet);
    }
}

// 关闭当前输入流，并把内部状态恢复为未打开状态。
void FfmpegDemuxer::Close() {
    if (format_context_ != NULL) {
        avformat_close_input(&format_context_);
        format_context_ = NULL;
    }
    video_stream_index_ = -1;
}

}  // namespace rknn_demo
