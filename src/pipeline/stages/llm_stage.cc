#include "pipeline/stages/llm_stage.h"

#include <exception>
#include <sstream>
#include <stdio.h>

#include "model/postprocess.h"

// 文件说明：
// 实现异步 LLM 线程，负责把抽样图片和 YOLO 结果发给云端视觉模型。

namespace rknn_demo {

namespace {

std::string BuildDetectionsSummary(const object_detect_result_list &detections) {
    if (detections.count <= 0) {
        return "YOLO detections: none.";
    }

    std::ostringstream summary;
    summary << "YOLO detections:";
    const int max_items = detections.count < 10 ? detections.count : 10;
    for (int i = 0; i < max_items; ++i) {
        const object_detect_result &det = detections.results[i];
        summary << ' '
                << coco_cls_to_name(det.cls_id)
                << '(' << det.prop << ')'
                << '[' << det.box.left
                << ',' << det.box.top
                << ',' << det.box.right
                << ',' << det.box.bottom
                << ']';
        if (i + 1 < max_items) {
            summary << ';';
        }
    }

    if (detections.count > max_items) {
        summary << " ...";
    }

    return summary.str();
}

std::string BuildVisionPrompt(const std::string &base_prompt,
                              const object_detect_result_list &detections) {
    std::ostringstream prompt;
    prompt << base_prompt;
    if (!base_prompt.empty() && base_prompt[base_prompt.size() - 1] != '\n') {
        prompt << '\n';
    }
    prompt << "请结合图像与下面的 YOLO 检测结果，用简洁中文描述当前场景，并指出值得关注的对象或风险。"
           << '\n'
           << BuildDetectionsSummary(detections);
    return prompt.str();
}

}  // namespace

LlmStage::LlmStage(const AppOptions &options,
                   PipelineQueues *queues,
                   PipelineRuntime *runtime)
    : options_(options),
      queues_(queues),
      runtime_(runtime),
      initialized_(false) {}

bool LlmStage::Init() {
    std::string error_message;
    if (!LLMApi::LLMApi::LoadVisionConfig(options_.llm_config_path,
                                          &vision_config_,
                                          &error_message)) {
        printf("llm: failed to load config from %s: %s\n",
               options_.llm_config_path.c_str(),
               error_message.c_str());
        return false;
    }

    client_ = LLMApi::LLMApi(vision_config_.api_key, vision_config_.base_url);
    initialized_ = true;
    return true;
}

void LlmStage::Run() {
    if (!initialized_) {
        printf("llm: stage not initialized, skip worker loop\n");
        return;
    }

    LlmRequestPacket request;
    while (!runtime_->stop_requested.load() && queues_->llm_queue().pop(&request)) {
        try {
            const std::string prompt = BuildVisionPrompt(vision_config_.prompt,
                                                         request.detections);
            const std::string reply = client_.AnalyzeVision(vision_config_.model,
                                                            prompt,
                                                            request.image_path,
                                                            vision_config_.image_detail);
            runtime_->stats.llm_completed.fetch_add(1, std::memory_order_relaxed);
            printf("llm: frame=%llu reply=%s\n",
                   static_cast<unsigned long long>(request.frame_id),
                   reply.c_str());
        } catch (const std::exception &ex) {
            runtime_->stats.llm_failed.fetch_add(1, std::memory_order_relaxed);
            printf("llm: frame=%llu request failed: %s\n",
                   static_cast<unsigned long long>(request.frame_id),
                   ex.what());
        }
    }
}

}  // namespace rknn_demo
