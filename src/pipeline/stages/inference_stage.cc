#include "pipeline/stages/inference_stage.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <string>

#include "image_utils.h"
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

bool EnsureDirectoryExists(const std::string &path) {
    if (path.empty() || path == ".") {
        return true;
    }

    std::string partial;
    if (path[0] == '/') {
        partial = "/";
    }

    size_t start = (path[0] == '/') ? 1 : 0;
    while (start <= path.size()) {
        size_t slash = path.find('/', start);
        std::string segment = path.substr(start, slash == std::string::npos
                                                     ? std::string::npos
                                                     : slash - start);
        if (!segment.empty()) {
            if (!partial.empty() && partial[partial.size() - 1] != '/') {
                partial += '/';
            }
            partial += segment;
            if (mkdir(partial.c_str(), 0755) != 0 && errno != EEXIST) {
                return false;
            }
        }

        if (slash == std::string::npos) {
            break;
        }
        start = slash + 1;
    }

    return true;
}

bool ShouldSampleFrameForLlm(const AppOptions &options,
                             PipelineRuntime *runtime,
                             const ResultPacket &result) {
    if (!options.enable_llm || result.detections.count <= 0) {
        return false;
    }
    if (options.llm_sample_every > 1 &&
        (result.frame_id % static_cast<uint64_t>(options.llm_sample_every)) != 0) {
        return false;
    }

    if (options.llm_min_interval_ms <= 0) {
        return true;
    }

    const uint64_t now_ms = static_cast<uint64_t>(NowMs());
    const uint64_t min_interval_ms = static_cast<uint64_t>(options.llm_min_interval_ms);
    while (true) {
        uint64_t last_ms = runtime->last_llm_submit_ms.load(std::memory_order_relaxed);
        if (last_ms != 0 && now_ms >= last_ms && now_ms - last_ms < min_interval_ms) {
            return false;
        }
        if (runtime->last_llm_submit_ms.compare_exchange_weak(last_ms, now_ms)) {
            return true;
        }
    }
}

bool SaveLlmSampleImage(const AppOptions &options,
                        const image_buffer_t &src_image,
                        uint64_t frame_id,
                        std::string *image_path) {
    if (image_path == NULL) {
        return false;
    }
    if (!EnsureDirectoryExists(options.llm_output_dir)) {
        return false;
    }

    image_buffer_t sample_image;
    memset(&sample_image, 0, sizeof(sample_image));
    sample_image.width = options.llm_image_width;
    sample_image.height = options.llm_image_height;
    sample_image.width_stride = options.llm_image_width;
    sample_image.height_stride = options.llm_image_height;
    sample_image.format = IMAGE_FORMAT_RGB888;
    sample_image.fd = -1;

    image_buffer_t src_copy = src_image;
    int ret = convert_image(&src_copy, &sample_image, NULL, NULL, 0);
    if (ret != 0) {
        free(sample_image.virt_addr);
        sample_image.virt_addr = NULL;
        return false;
    }

    char filename[128];
    snprintf(filename, sizeof(filename), "frame_%012llu.jpg",
             static_cast<unsigned long long>(frame_id));

    std::string path = options.llm_output_dir;
    if (!path.empty() && path[path.size() - 1] != '/') {
        path += '/';
    }
    path += filename;

    ret = write_image(path.c_str(), &sample_image);
    free(sample_image.virt_addr);
    sample_image.virt_addr = NULL;
    if (ret == 0) {
        *image_path = path;
        return true;
    }
    return false;
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

        if (ShouldSampleFrameForLlm(options_, runtime_, result)) {
            std::string sample_image_path;
            if (SaveLlmSampleImage(options_, src_image, result.frame_id, &sample_image_path)) {
                LlmRequestPacket llm_request;
                llm_request.frame_id = result.frame_id;
                llm_request.capture_ts_ms = result.capture_ts_ms;
                llm_request.infer_done_ts_ms = result.infer_done_ts_ms;
                llm_request.detections = result.detections;
                llm_request.image_path = sample_image_path;
                llm_request.delete_image_on_destroy = true;

                bool llm_dropped = false;
                if (queues_->llm_queue().push(std::move(llm_request), &llm_dropped)) {
                    runtime_->stats.llm_submitted.fetch_add(1, std::memory_order_relaxed);
                    if (llm_dropped) {
                        printf("infer[%d]: llm queue dropped oldest sample before frame=%llu\n",
                               worker_id_,
                               static_cast<unsigned long long>(result.frame_id));
                    }
                }
            } else {
                printf("infer[%d]: failed to prepare llm sample for frame=%llu\n",
                       worker_id_,
                       static_cast<unsigned long long>(result.frame_id));
            }
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
        queues_->llm_queue().stop();
    }
}

}  // namespace rknn_demo
