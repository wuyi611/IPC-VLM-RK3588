#include "media/mpp_decoder.h"

#include <stdio.h>
#include <string.h>

#include <rockchip/mpp_err.h>
#include <rockchip/mpp_packet.h>
#include <rockchip/mpp_task.h>

// 文件说明：
// 实现基于纯 MPP 的硬件解码器封装，直接输出 DMA-BUF 帧供 RGA / RKNN 使用。

namespace rknn_demo {

namespace {

const int kDecoderBufferCount = 24;  // 解码输出缓冲组允许缓存的最大帧数。

// 把 MPP 帧格式映射成项目内部统一使用的图像格式枚举。
image_format_t MapMppFrameFormat(MppFrameFormat format) {
    switch (format & MPP_FRAME_FMT_MASK) {
    case MPP_FMT_YUV420SP:
        return IMAGE_FORMAT_YUV420SP_NV12;
    case MPP_FMT_YUV420SP_VU:
        return IMAGE_FORMAT_YUV420SP_NV21;
    default:
        return IMAGE_FORMAT_GRAY8;
    }
}

// 判断当前 MPP 返回值是否属于“暂时失败，可稍后重试”的情况。
bool IsRetryableMppRet(MPP_RET ret) {
    return ret == MPP_ERR_BUFFER_FULL || ret == MPP_NOK || ret == MPP_ERR_TIMEOUT;
}

}  // namespace

// 构造一个尚未初始化的解码器实例。
MppDecoder::MppDecoder()
    : mpp_ctx_(NULL),
      mpp_api_(NULL),
      output_buffer_group_(NULL),
      initialized_(false),
      draining_(false),
      eos_reached_(false),
      fatal_error_(false),
      output_group_attached_(false) {}

// 析构时统一回收 MPP 资源。
MppDecoder::~MppDecoder() {
    Reset();
}

// 根据流参数创建并初始化一套新的 MPP 解码器。
bool MppDecoder::Init(const MediaStreamInfo &stream_info) {
    Reset();
    stream_info_ = stream_info;

    MppCodingType coding_type = GetCodingType(stream_info.codec_id);  // 当前输入流对应的 MPP 编码类型。
    if (coding_type == MPP_VIDEO_CodingUnused) {
        printf("unsupported codec for pure mpp decoder, codec_id=%d\n", stream_info.codec_id);
        return false;
    }

    MPP_RET ret = mpp_create(&mpp_ctx_, &mpp_api_);  // 创建 MPP 上下文和 API 接口。
    if (ret != MPP_OK) {
        printf("mpp_create failed, ret=%d\n", ret);
        Reset();
        return false;
    }

    if (!ConfigureParser()) {
        Reset();
        return false;
    }

    ret = mpp_init(mpp_ctx_, MPP_CTX_DEC, coding_type);
    if (ret != MPP_OK) {
        printf("mpp_init failed, ret=%d\n", ret);
        Reset();
        return false;
    }

    if (!ConfigureOutputFormat()) {
        Reset();
        return false;
    }

    initialized_ = true;
    return true;
}

// 把一份压缩视频包送入 MPP 解码器。
MppDecoder::FeedStatus MppDecoder::SendPacket(const EncodedPacket &packet) {
    if (!initialized_ || fatal_error_) {
        return kFeedError;
    }

    if (packet.packet.data == NULL || packet.packet.size <= 0) {
        return kFeedRetry;
    }

    MppPacket mpp_packet = NULL;  // 交给 MPP 的输入包句柄。
    MPP_RET ret = mpp_packet_init(&mpp_packet,
                                  packet.packet.data,
                                  static_cast<size_t>(packet.packet.size));
    if (ret != MPP_OK || mpp_packet == NULL) {
        printf("mpp_packet_init failed, ret=%d\n", ret);
        fatal_error_ = true;
        return kFeedError;
    }

    mpp_packet_set_pos(mpp_packet, packet.packet.data);
    mpp_packet_set_length(mpp_packet, static_cast<size_t>(packet.packet.size));

    int64_t pts = packet.packet.pts;  // 送入 MPP 的时间戳键，用于后续输出帧回查元信息。
    if (pts == AV_NOPTS_VALUE) {
        pts = static_cast<int64_t>(packet.frame_id);
    }
    mpp_packet_set_pts(mpp_packet, pts);
    packet_meta_by_pts_[pts] = PacketMeta{packet.frame_id, packet.capture_ts_ms};

    FeedStatus status = SendMppPacket(mpp_packet);  // 记录本次送包结果。
    mpp_packet_deinit(&mpp_packet);
    return status;
}

// 发送一份 EOS 包，通知解码器进入排空流程。
MppDecoder::FeedStatus MppDecoder::SendEos() {
    if (!initialized_ || fatal_error_) {
        return kFeedError;
    }

    MppPacket eos_packet = NULL;  // 专门用于标记流结束的空包。
    MPP_RET ret = mpp_packet_init(&eos_packet, NULL, 0);
    if (ret != MPP_OK || eos_packet == NULL) {
        printf("mpp_packet_init eos failed, ret=%d\n", ret);
        fatal_error_ = true;
        return kFeedError;
    }

    mpp_packet_set_eos(eos_packet);
    FeedStatus status = SendMppPacket(eos_packet);  // 记录 EOS 包送入状态。
    if (status == kFeedOk) {
        draining_ = true;
    }
    mpp_packet_deinit(&eos_packet);
    return status;
}

// 从 MPP 中拉取一帧解码结果，并回填项目内部元信息。
bool MppDecoder::ReceiveFrame(DecodedPacket *decoded_packet) {
    if (!initialized_ || fatal_error_ || decoded_packet == NULL) {
        return false;
    }

    RK_S64 timeout = draining_ ? MPP_TIMEOUT_BLOCK : MPP_TIMEOUT_NON_BLOCK;  // 排空阶段阻塞等待，其余阶段非阻塞轮询。
    MPP_RET ret = mpp_api_->control(
        mpp_ctx_,
        MPP_SET_OUTPUT_TIMEOUT,
        reinterpret_cast<MppParam>(&timeout));
    if (ret != MPP_OK) {
        printf("mpp control output timeout failed, ret=%d\n", ret);
        fatal_error_ = true;
        return false;
    }

    MppFrame frame = NULL;  // 保存 MPP 返回的原始解码帧。
    ret = mpp_api_->decode_get_frame(mpp_ctx_, &frame);
    if (ret != MPP_OK) {
        if (!IsRetryableMppRet(ret)) {
            printf("decode_get_frame failed, ret=%d\n", ret);
            fatal_error_ = true;
        }
        return false;
    }

    if (frame == NULL) {
        return false;
    }

    if (mpp_frame_get_info_change(frame)) {
        int width = static_cast<int>(mpp_frame_get_width(frame));  // 解码器上报的新帧宽度。
        int height = static_cast<int>(mpp_frame_get_height(frame));  // 解码器上报的新帧高度。
        if (!EnsureOutputBufferGroup(width, height)) {
            mpp_frame_deinit(&frame);
            fatal_error_ = true;
            return false;
        }
        ret = mpp_api_->control(mpp_ctx_, MPP_DEC_SET_INFO_CHANGE_READY, NULL);
        if (ret != MPP_OK) {
            printf("MPP_DEC_SET_INFO_CHANGE_READY failed, ret=%d\n", ret);
            mpp_frame_deinit(&frame);
            fatal_error_ = true;
            return false;
        }
        mpp_frame_deinit(&frame);
        return false;
    }

    if (mpp_frame_get_discard(frame) || mpp_frame_get_errinfo(frame)) {
        mpp_frame_deinit(&frame);
        return false;
    }

    MppBuffer buffer = mpp_frame_get_buffer(frame);  // 当前解码帧绑定的底层缓冲对象。
    if (buffer == NULL) {
        eos_reached_ = mpp_frame_get_eos(frame) != 0;
        mpp_frame_deinit(&frame);
        return false;
    }

    image_buffer_t image;  // 提供给后续 RGA / RKNN 使用的图像描述。
    if (!BuildImageBufferFromMppFrame(frame, &image)) {
        printf("unsupported mpp frame format=%d\n", mpp_frame_get_fmt(frame));
        mpp_frame_deinit(&frame);
        fatal_error_ = true;
        return false;
    }

    int64_t pts = mpp_frame_get_pts(frame);  // 输出帧的时间戳键。
    PacketMeta meta = {0, 0.0};  // 当查不到历史记录时使用的兜底元信息。
    std::unordered_map<int64_t, PacketMeta>::iterator it =
        packet_meta_by_pts_.find(pts);  // 在缓存表中查找与当前输出帧对应的送包记录。
    if (it != packet_meta_by_pts_.end()) {
        meta = it->second;
        packet_meta_by_pts_.erase(it);
    }

    decoded_packet->frame_id = meta.frame_id;
    decoded_packet->capture_ts_ms = meta.capture_ts_ms;
    decoded_packet->decoded_frame.Reset(frame, &image);

    if (mpp_frame_get_eos(frame)) {
        eos_reached_ = true;
    }
    return true;
}

// 判断排空阶段是否已经真正走到最后一帧。
bool MppDecoder::Drained() const {
    return draining_ && eos_reached_;
}

// 查询解码器内部是否已经进入致命错误状态。
bool MppDecoder::HasFatalError() const {
    return fatal_error_;
}

// 释放 MPP 上下文、缓冲组和缓存的元信息。
void MppDecoder::Reset() {
    packet_meta_by_pts_.clear();
    initialized_ = false;
    draining_ = false;
    eos_reached_ = false;
    fatal_error_ = false;
    output_group_attached_ = false;

    if (output_buffer_group_ != NULL) {
        mpp_buffer_group_put(output_buffer_group_);
        output_buffer_group_ = NULL;
    }

    if (mpp_ctx_ != NULL) {
        mpp_destroy(mpp_ctx_);
        mpp_ctx_ = NULL;
        mpp_api_ = NULL;
    }
}

// 配置 parser 的拆包模式和快速模式。
bool MppDecoder::ConfigureParser() const {
    int split_mode = 1;  // 打开 parser 拆包，让 MPP 自己切分输入码流。
    MPP_RET ret = mpp_api_->control(
        mpp_ctx_,
        MPP_DEC_SET_PARSER_SPLIT_MODE,
        reinterpret_cast<MppParam>(&split_mode));
    if (ret != MPP_OK) {
        printf("MPP_DEC_SET_PARSER_SPLIT_MODE failed, ret=%d\n", ret);
        return false;
    }

    int fast_mode = 1;  // 打开 parser 快速模式，减少额外解析开销。
    ret = mpp_api_->control(
        mpp_ctx_,
        MPP_DEC_SET_PARSER_FAST_MODE,
        reinterpret_cast<MppParam>(&fast_mode));
    if (ret != MPP_OK) {
        printf("MPP_DEC_SET_PARSER_FAST_MODE failed, ret=%d\n", ret);
        return false;
    }

    return true;
}

// 把解码器输出格式固定为 RGA / RKNN 更容易消费的 NV12。
bool MppDecoder::ConfigureOutputFormat() const {
    MppFrameFormat output_format = MPP_FMT_YUV420SP;  // 目标输出格式固定为 NV12。
    MPP_RET ret = mpp_api_->control(
        mpp_ctx_,
        MPP_DEC_SET_OUTPUT_FORMAT,
        reinterpret_cast<MppParam>(&output_format));
    if (ret != MPP_OK) {
        printf("MPP_DEC_SET_OUTPUT_FORMAT failed, ret=%d\n", ret);
        return false;
    }
    return true;
}

// 为解码器准备并绑定外部输出缓冲组。
bool MppDecoder::EnsureOutputBufferGroup(int width, int height) {
    (void)width;
    (void)height;

    if (!output_group_attached_) {
        if (output_buffer_group_ == NULL) {
            MPP_RET ret = mpp_buffer_group_get_internal(
                &output_buffer_group_,
                MPP_BUFFER_TYPE_DRM);  // 申请可导出 DMA-BUF 的 DRM 缓冲组。
            if (ret != MPP_OK) {
                printf("mpp_buffer_group_get_internal output failed, ret=%d\n", ret);
                return false;
            }
            ret = mpp_buffer_group_limit_config(
                output_buffer_group_,
                0,
                kDecoderBufferCount);  // 限制缓冲组里最多缓存的帧数。
            if (ret != MPP_OK) {
                printf("mpp_buffer_group_limit_config failed, ret=%d\n", ret);
            }
        }

        MPP_RET ret = mpp_api_->control(
            mpp_ctx_,
            MPP_DEC_SET_EXT_BUF_GROUP,
            reinterpret_cast<MppParam>(output_buffer_group_));  // 把外部缓冲组挂到当前解码器上。
        if (ret != MPP_OK) {
            printf("MPP_DEC_SET_EXT_BUF_GROUP failed, ret=%d\n", ret);
            return false;
        }
        output_group_attached_ = true;
    }

    return true;
}

// 对 `decode_put_packet` 的结果做统一封装。
MppDecoder::FeedStatus MppDecoder::SendMppPacket(MppPacket packet) {
    MPP_RET ret = mpp_api_->decode_put_packet(mpp_ctx_, packet);  // 真正向 MPP 投递输入包。
    if (ret == MPP_OK) {
        return kFeedOk;
    }
    if (IsRetryableMppRet(ret)) {
        return kFeedRetry;
    }

    printf("decode_put_packet failed, ret=%d\n", ret);
    fatal_error_ = true;
    return kFeedError;
}

// 把 FFmpeg 的 codec id 映射到 MPP 能识别的编码类型。
MppCodingType MppDecoder::GetCodingType(AVCodecID codec_id) const {
    switch (codec_id) {
    case AV_CODEC_ID_H263:
        return MPP_VIDEO_CodingH263;
    case AV_CODEC_ID_H264:
        return MPP_VIDEO_CodingAVC;
    case AV_CODEC_ID_HEVC:
        return MPP_VIDEO_CodingHEVC;
    case AV_CODEC_ID_AV1:
        return MPP_VIDEO_CodingAV1;
    case AV_CODEC_ID_VP8:
        return MPP_VIDEO_CodingVP8;
    case AV_CODEC_ID_VP9:
        return MPP_VIDEO_CodingVP9;
    case AV_CODEC_ID_MPEG1VIDEO:
    case AV_CODEC_ID_MPEG2VIDEO:
        return MPP_VIDEO_CodingMPEG2;
    case AV_CODEC_ID_MPEG4:
        return MPP_VIDEO_CodingMPEG4;
    case AV_CODEC_ID_MJPEG:
        return MPP_VIDEO_CodingMJPEG;
    default:
        return MPP_VIDEO_CodingUnused;
    }
}

// 从 `MppFrame` 中构造项目内部的图像缓冲描述。
bool MppDecoder::BuildImageBufferFromMppFrame(MppFrame frame, image_buffer_t *image) const {
    if (frame == NULL || image == NULL) {
        return false;
    }

    MppBuffer buffer = mpp_frame_get_buffer(frame);  // 帧对应的底层 MPP 缓冲。
    if (buffer == NULL) {
        return false;
    }

    image_format_t image_format =
        MapMppFrameFormat(mpp_frame_get_fmt(frame));  // 项目内部使用的图像格式描述。
    if (image_format != IMAGE_FORMAT_YUV420SP_NV12 &&
        image_format != IMAGE_FORMAT_YUV420SP_NV21) {
        return false;
    }

    memset(image, 0, sizeof(*image));
    image->width = static_cast<int>(mpp_frame_get_width(frame));
    image->height = static_cast<int>(mpp_frame_get_height(frame));
    image->width_stride = static_cast<int>(mpp_frame_get_hor_stride(frame));
    image->height_stride = static_cast<int>(mpp_frame_get_ver_stride(frame));
    image->format = image_format;
    image->virt_addr = static_cast<unsigned char *>(mpp_buffer_get_ptr(buffer));
    image->size = static_cast<int>(mpp_buffer_get_size(buffer));
    image->fd = mpp_buffer_get_fd(buffer);
    return image->fd >= 0;
}

}  // namespace rknn_demo
