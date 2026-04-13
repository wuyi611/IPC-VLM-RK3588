#ifndef RKNN_YOLOV5_DEMO_PIPELINE_STAGES_CAPTURE_STAGE_H_
#define RKNN_YOLOV5_DEMO_PIPELINE_STAGES_CAPTURE_STAGE_H_

#include <memory>

#include "app/app_options.h"
#include "media/ffmpeg_demuxer.h"
#include "pipeline/pipeline_queues.h"
#include "pipeline/pipeline_types.h"

// 文件说明：
// 定义采集阶段，负责从输入源读取压缩视频包并送入第一段队列。

namespace rknn_demo {

// `CaptureStage` 负责整条流水线的最前端输入。
// 它只关心“从输入源取到压缩包并投递”，不参与解码、推理和显示。
class CaptureStage {
public:
    // 构造采集阶段对象。
    // `demuxer` 的所有权会被转移进来，后续由采集阶段独占使用。
    CaptureStage(const AppOptions &options,
                 std::unique_ptr<FfmpegDemuxer> demuxer,
                 PipelineQueues *queues,
                 PipelineRuntime *runtime);

    // 采集线程主循环。
    // 持续读取压缩包，生成帧编号与时间戳，并写入 capture 队列。
    void Run();

private:
    // 启动参数引用。
    const AppOptions &options_;

    // 实际负责读取输入流的 demuxer。
    std::unique_ptr<FfmpegDemuxer> demuxer_;

    // 各阶段共享的队列集合。
    PipelineQueues *queues_;

    // 全局运行时状态。
    PipelineRuntime *runtime_;
};

}  // namespace rknn_demo

#endif  // RKNN_YOLOV5_DEMO_PIPELINE_STAGES_CAPTURE_STAGE_H_
