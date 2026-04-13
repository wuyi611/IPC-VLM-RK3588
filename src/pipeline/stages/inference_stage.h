#ifndef RKNN_YOLOV5_DEMO_PIPELINE_STAGES_INFERENCE_STAGE_H_
#define RKNN_YOLOV5_DEMO_PIPELINE_STAGES_INFERENCE_STAGE_H_

#include "app/app_options.h"
#include "model/model_context.h"
#include "pipeline/pipeline_queues.h"
#include "pipeline/pipeline_types.h"

// 文件说明：
// 定义推理阶段，一个实例对应一个独立的 RKNN 推理 worker。

namespace rknn_demo {

// `InferenceStage` 表示一条推理工作线程。
// 每个实例都独占一个模型上下文，从解码队列取帧并向渲染队列输出结果。
class InferenceStage {
public:
    // 构造一个推理 worker。
    InferenceStage(int worker_id,
                   const AppOptions &options,
                   rknn_app_context_t *app_context,
                   PipelineQueues *queues,
                   PipelineRuntime *runtime);

    // 推理线程主循环。
    void Run();

private:
    // 当前推理 worker 的编号。
    int worker_id_;

    // 启动参数引用。
    const AppOptions &options_;

    // 当前 worker 独占的 RKNN 模型上下文。
    rknn_app_context_t *app_context_;

    // 各阶段共享的队列集合。
    PipelineQueues *queues_;

    // 全局运行时状态。
    PipelineRuntime *runtime_;
};

}  // namespace rknn_demo

#endif  // RKNN_YOLOV5_DEMO_PIPELINE_STAGES_INFERENCE_STAGE_H_
