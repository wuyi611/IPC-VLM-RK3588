#ifndef RKNN_YOLOV5_DEMO_PIPELINE_STAGES_LLM_STAGE_H_
#define RKNN_YOLOV5_DEMO_PIPELINE_STAGES_LLM_STAGE_H_

#include <string>

#include "app/app_options.h"
#include "cloud/providers/LLMApi.h"
#include "pipeline/pipeline_queues.h"
#include "pipeline/pipeline_types.h"

// 文件说明：
// 定义异步 LLM 阶段，负责消费抽样图片并向云端视觉模型发起请求。

namespace rknn_demo {

class LlmStage {
public:
    LlmStage(const AppOptions &options,
             PipelineQueues *queues,
             PipelineRuntime *runtime);

    // 加载云端接口配置并初始化客户端。
    bool Init();

    // LLM 线程主循环。
    void Run();

private:
    const AppOptions &options_;
    PipelineQueues *queues_;
    PipelineRuntime *runtime_;
    bool initialized_;
    LLMApi::VisionConfig vision_config_;
    LLMApi::LLMApi client_;
};

}  // namespace rknn_demo

#endif  // RKNN_YOLOV5_DEMO_PIPELINE_STAGES_LLM_STAGE_H_
