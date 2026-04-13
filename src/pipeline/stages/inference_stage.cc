#include "pipeline/stages/inference_stage.h"

#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#include "model/yolov5_model.h"
#include "model/yolov8_model.h"
#include "pipeline/pipeline_utils.h"

// 文件说明：
// 实现推理线程逻辑，负责预处理、RKNN 推理和结果投递。

namespace rknn_demo {

namespace {

// 当前工程通过命令行选项决定是否走 YOLOv8 分支。
bool UseYolov8Model(const AppOptions &options) {
    return options.model_type == "yolov8";
}

}  // namespace

// 保存某个推理 worker 独占的模型上下文及共享运行对象。
InferenceStage::InferenceStage(int worker_id,
                               const AppOptions &options,
                               rknn_app_context_t *app_context,
                               PipelineQueues *queues,
                               PipelineRuntime *runtime)
    : worker_id_(worker_id),
      options_(options),
      app_context_(app_context),
      queues_(queues),
      runtime_(runtime) {}

// 消费解码帧，执行一次完整的预处理、NPU 推理和后处理流程。
void InferenceStage::Run() {
    DecodedPacket decoded;  // 当前从解码队列取出的帧。
    bool first_result_logged = false;  // 控制首个结果日志只打印一次。
    bool build_image_error_logged = false;  // 控制无效 DMA-BUF 日志只打印一次。

    while (!runtime_->stop_requested.load() && queues_->decode_queue().pop(&decoded)) {
        if (decoded.decoded_frame.empty()) {
            continue;
        }

        // 直接复用纯 MPP 解码输出的 DMA-BUF 图像描述，避免再走 CPU 帧转换。
        image_buffer_t src_image = decoded.decoded_frame.image;
        if (src_image.fd < 0) {
            if (!build_image_error_logged) {
                printf("infer[%d]: invalid decoded dma-buf frame, fd=%d format=%d\n",
                       worker_id_,
                       src_image.fd,
                       src_image.format);
                build_image_error_logged = true;
            }
            continue;
        }

        ResultPacket result;  // 当前要投递给渲染阶段的结果包。
        memset(&result.detections, 0, sizeof(result.detections));
        result.frame_id = decoded.frame_id;
        result.capture_ts_ms = decoded.capture_ts_ms;
        result.decode_done_ts_ms = decoded.decode_done_ts_ms;
        result.infer_start_ts_ms = NowMs();

        struct timeval start_time;  // 推理开始前的时间点。
        struct timeval stop_time;  // 推理结束后的时间点。
        gettimeofday(&start_time, NULL);

        InferenceProfile profile;  // 当前推理内部各子阶段的性能统计。
        int ret = UseYolov8Model(options_)
                      ? inference_yolov8_model(app_context_, &src_image, &result.detections, &profile)
                      : inference_yolov5_model(app_context_, &src_image, &result.detections, &profile);  // 本次推理调用的返回值。

        gettimeofday(&stop_time, NULL);
        result.infer_ms = (GetUs(stop_time) - GetUs(start_time)) / 1000.0;
        result.preprocess_ms = profile.preprocess_ms;
        result.npu_ms = profile.npu_ms;
        result.postprocess_ms = profile.postprocess_ms;
        result.infer_done_ts_ms = NowMs();

        if (ret != 0) {
            printf("worker %d inference failed, ret=%d\n", worker_id_, ret);
            continue;
        }

        // 如果启用了显示，就把原始 DMA-BUF 解码帧所有权一并传给渲染阶段。
        if (options_.enable_display) {
            result.display_frame = std::move(decoded.decoded_frame);
        }

        bool dropped = false;  // 记录本次写入是否因为队列已满而丢掉了旧结果。
        if (!queues_->render_queue().push(std::move(result), &dropped)) {
            break;
        }

        runtime_->stats.inferred.fetch_add(1, std::memory_order_relaxed);
        if (dropped) {
            runtime_->stats.dropped_render_queue.fetch_add(1, std::memory_order_relaxed);
        }

        if (!first_result_logged) {
            printf("infer[%d]: first result ready, frame=%llu detections=%d infer=%.2fms\n",
                   worker_id_,
                   static_cast<unsigned long long>(result.frame_id),
                   result.detections.count,
                   result.infer_ms);
            first_result_logged = true;
        }
    }

    // 最后退出的那个推理线程负责关闭渲染队列，
    // 这样渲染线程就能自然从阻塞等待中退出。
    if (runtime_->active_infer_workers.fetch_sub(1) == 1) {
        queues_->render_queue().stop();
    }
}

}  // namespace rknn_demo
