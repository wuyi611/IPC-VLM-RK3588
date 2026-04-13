#ifndef RKNN_YOLOV5_DEMO_MEDIA_MPP_DECODER_H_
#define RKNN_YOLOV5_DEMO_MEDIA_MPP_DECODER_H_

#include <stdint.h>

#include <rockchip/mpp_buffer.h>
#include <rockchip/rk_mpi.h>

#include <unordered_map>

#include "media/ffmpeg_demuxer.h"
#include "pipeline/pipeline_types.h"

// 文件说明：
// 定义基于纯 MPP 的硬件解码器封装，输出可直接交给 RGA / RKNN 的 DMA-BUF 帧。

namespace rknn_demo {

// `MppDecoder` 负责把压缩视频包送入 Rockchip MPP，并取回解码后的图像帧。
class MppDecoder {
public:
    // 描述一次送包动作的结果，供上游决定是否继续重试。
    enum FeedStatus {
        kFeedOk,     // 数据包已成功送入 MPP。
        kFeedRetry,  // 解码器暂时繁忙，调用方稍后可重试。
        kFeedError,  // 出现不可恢复错误，需要停止继续送包。
    };

    // 构造一个尚未初始化的 MPP 解码器。
    MppDecoder();

    // 析构时释放解码器上下文和输出缓冲组。
    ~MppDecoder();

    // 根据输入流信息创建并配置解码器实例。
    bool Init(const MediaStreamInfo &stream_info);

    // 向解码器送入一份压缩视频包。
    FeedStatus SendPacket(const EncodedPacket &packet);

    // 向解码器发送 EOS 包，通知后续不会再有新数据。
    FeedStatus SendEos();

    // 从解码器尝试取出一帧已经解码完成的图像。
    bool ReceiveFrame(DecodedPacket *decoded_packet);

    // 判断解码器是否已经在排空阶段收到了最后一帧。
    bool Drained() const;

    // 判断内部是否已经出现不可恢复的 MPP 错误。
    bool HasFatalError() const;

    // 释放所有 MPP 资源，并把对象恢复到未初始化状态。
    void Reset();

private:
    // 保存送包时记录的元信息，便于按 PTS 回填到输出帧。
    struct PacketMeta {
        uint64_t frame_id;      // 项目内部生成的顺序帧编号。
        double capture_ts_ms;   // 原始采集时间戳，单位毫秒。
    };

    // 打开 parser 拆包与快速解析模式，提升硬解稳定性。
    bool ConfigureParser() const;

    // 配置解码输出像素格式。
    bool ConfigureOutputFormat() const;

    // 确保 MPP 已经绑定好外部输出缓冲组。
    bool EnsureOutputBufferGroup(int width, int height);

    // 把一个已经构造好的 `MppPacket` 送入解码器。
    FeedStatus SendMppPacket(MppPacket packet);

    // 把 FFmpeg codec id 转换成 MPP 识别的编码类型。
    MppCodingType GetCodingType(AVCodecID codec_id) const;

    // 从 `MppFrame` 中提取出可供后续模块使用的 `image_buffer_t` 描述。
    bool BuildImageBufferFromMppFrame(MppFrame frame, image_buffer_t *image) const;

    MppCtx mpp_ctx_;  // MPP 解码上下文句柄。
    MppApi *mpp_api_;  // MPP 提供的控制与取帧接口表。
    MppBufferGroup output_buffer_group_;  // 解码输出使用的外部缓冲组。
    MediaStreamInfo stream_info_;  // 当前解码器绑定的输入流参数。
    bool initialized_;  // 标记解码器是否已成功初始化。
    bool draining_;  // 标记是否已经进入 EOS 排空阶段。
    bool eos_reached_;  // 标记是否已经收到带 EOS 的输出帧。
    bool fatal_error_;  // 标记是否发生了不可恢复错误。
    bool output_group_attached_;  // 标记外部缓冲组是否已经绑定给解码器。
    std::unordered_map<int64_t, PacketMeta> packet_meta_by_pts_;  // 按 PTS 缓存送包元信息。
};

}  // namespace rknn_demo

#endif  // RKNN_YOLOV5_DEMO_MEDIA_MPP_DECODER_H_
