#ifndef RKNN_YOLOV5_DEMO_PIPELINE_STAGES_RENDER_STAGE_H_
#define RKNN_YOLOV5_DEMO_PIPELINE_STAGES_RENDER_STAGE_H_

#include "app/app_options.h"
#include "pipeline/pipeline_queues.h"
#include "pipeline/pipeline_types.h"

// 文件说明：
// 定义渲染阶段，负责显示、统计打印和最终消费推理结果。

namespace rknn_demo {

// `RenderStage` 是流水线最后一个阶段。
// 即使关闭了本地显示，它仍然负责统计输出和消费推理结果。
class RenderStage {
public:
    // 构造渲染阶段对象。
    RenderStage(const AppOptions &options,
                PipelineQueues *queues,
                PipelineRuntime *runtime);

    // 渲染线程主循环。
    void Run();

private:
    // 启动参数引用。
    const AppOptions &options_;

    // 各阶段共享的队列集合。
    PipelineQueues *queues_;

    // 全局运行时状态。
    PipelineRuntime *runtime_;
};

}  // namespace rknn_demo

#endif  // RKNN_YOLOV5_DEMO_PIPELINE_STAGES_RENDER_STAGE_H_
