#ifndef RKNN_YOLOV5_DEMO_PIPELINE_STAGES_DECODE_STAGE_H_
#define RKNN_YOLOV5_DEMO_PIPELINE_STAGES_DECODE_STAGE_H_

#include <chrono>

#include "media/ffmpeg_demuxer.h"
#include "media/mpp_decoder.h"
#include "pipeline/pipeline_queues.h"
#include "pipeline/pipeline_types.h"

// 文件说明：
// 定义解码阶段，负责把压缩码流送入纯 MPP 解码器并输出 DMA-BUF 视频帧。

namespace rknn_demo {

// `DecodeStage` 负责流水线中的硬件解码部分。
class DecodeStage {
public:
    // 使用探测得到的流信息构造解码阶段对象。
    DecodeStage(const MediaStreamInfo &stream_info,
                PipelineQueues *queues,
                PipelineRuntime *runtime);

    // 初始化底层纯 MPP 解码器。
    bool Init();

    // 解码线程主循环。
    // 从 capture 队列取压缩包，解码后写入 decode 队列。
    void Run();

private:
    // 输入流的编码参数。
    MediaStreamInfo stream_info_;

    // 实际负责与纯 MPP 交互的解码器封装。
    MppDecoder decoder_;

    // 解码器是否已经成功初始化。
    bool initialized_;

    // 各阶段共享的队列集合。
    PipelineQueues *queues_;

    // 全局运行时状态。
    PipelineRuntime *runtime_;
};

}  // namespace rknn_demo

#endif  // RKNN_YOLOV5_DEMO_PIPELINE_STAGES_DECODE_STAGE_H_
