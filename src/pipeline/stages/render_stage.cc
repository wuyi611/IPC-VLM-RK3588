#include "pipeline/stages/render_stage.h"

#include <stdio.h>
#include <stdlib.h>

#include <utility>

#include "display/drm_display.h"
#include "pipeline/pipeline_utils.h"

// 文件说明：
// 实现渲染线程逻辑，使用 DRM/KMS 负责本地显示并打印性能统计。

namespace rknn_demo {

// 保存渲染阶段依赖的参数、队列和运行时状态。
RenderStage::RenderStage(const AppOptions &options,
                         PipelineQueues *queues,
                         PipelineRuntime *runtime)
    : options_(options),
      queues_(queues),
      runtime_(runtime) {}

// 持续消费推理结果，负责显示、打印日志并做窗口级统计。
void RenderStage::Run() {
    bool display_enabled = options_.enable_display;  // 标记当前是否仍然启用本地显示。
    bool display_failure_logged = false;  // 控制显示失败日志只打印一次。
    DrmDisplay drm_display;  // 实际负责本地 DRM/KMS 显示输出的对象。

    if (display_enabled && !drm_display.Init()) {
        printf("render: drm display init failed, fallback to headless stats mode\n");
        display_enabled = false;
    }

    // 下面这一组局部变量用于累计一个统计窗口内的性能数据。
    uint64_t rendered_since_log = 0;  // 当前统计窗口内已消费的结果包数量。
    double decode_sum_since_log = 0.0;  // 当前窗口累计的解码耗时。
    double decode_queue_sum_since_log = 0.0;  // 当前窗口累计的解码后排队耗时。
    double infer_sum_since_log = 0.0;  // 当前窗口累计的整体推理耗时。
    double preprocess_sum_since_log = 0.0;  // 当前窗口累计的预处理耗时。
    double npu_sum_since_log = 0.0;  // 当前窗口累计的 NPU 执行耗时。
    double postprocess_sum_since_log = 0.0;  // 当前窗口累计的后处理耗时。
    double render_queue_sum_since_log = 0.0;  // 当前窗口累计的渲染前排队耗时。
    double latency_sum_since_log = 0.0;  // 当前窗口累计的端到端延迟。
    double last_log_ms = NowMs();  // 上一次打印统计信息时的时间点。

    ResultPacket result;  // 当前从渲染队列取出的推理结果。
    while (queues_->render_queue().pop(&result)) {
        double now = NowMs();  // 当前渲染线程处理该结果包的时间点。

        // 下面几个时间值分别对应不同阶段的排队/处理耗时。
        double decode_ms = result.decode_done_ts_ms - result.capture_ts_ms;  // 单帧解码耗时。
        double decode_queue_ms = result.infer_start_ts_ms - result.decode_done_ts_ms;  // 解码完成后等待推理的耗时。
        double render_queue_ms = now - result.infer_done_ts_ms;  // 推理完成后等待渲染的耗时。
        double latency_ms = now - result.capture_ts_ms;  // 从采集到当前时刻的端到端延迟。

        rendered_since_log++;
        decode_sum_since_log += decode_ms;
        decode_queue_sum_since_log += decode_queue_ms;
        infer_sum_since_log += result.infer_ms;
        preprocess_sum_since_log += result.preprocess_ms;
        npu_sum_since_log += result.npu_ms;
        postprocess_sum_since_log += result.postprocess_ms;
        render_queue_sum_since_log += render_queue_ms;
        latency_sum_since_log += latency_ms;
        runtime_->stats.rendered.fetch_add(1, std::memory_order_relaxed);

        LogDetectionsIfNeeded(options_, result);

        double elapsed_since_log = now - last_log_ms;  // 距离上次打印已过去的时间。
        double render_fps = elapsed_since_log > 0.0
                                ? (rendered_since_log * 1000.0 / elapsed_since_log)
                                : 0.0;  // 当前统计窗口内的渲染 FPS。

        if (display_enabled) {
            if (result.display_frame.empty() ||
                !drm_display.Present(std::move(result.display_frame),
                                     result.detections,
                                     render_fps,
                                     latency_ms)) {
                if (!display_failure_logged) {
                    printf("render: drm display present failed, disable local display and keep printing stats\n");
                    display_failure_logged = true;
                }
                display_enabled = false;
            }
        }

        // 达到统计窗口大小后，打印一次平均性能信息并清零累计值。
        if (rendered_since_log >= static_cast<uint64_t>(options_.stats_interval)) {
            double avg_decode = decode_sum_since_log / rendered_since_log;  // 平均解码耗时。
            double avg_decode_queue = decode_queue_sum_since_log / rendered_since_log;  // 平均解码后等待推理耗时。
            double avg_preprocess = preprocess_sum_since_log / rendered_since_log;  // 平均预处理耗时。
            double avg_npu = npu_sum_since_log / rendered_since_log;  // 平均 NPU 耗时。
            double avg_postprocess = postprocess_sum_since_log / rendered_since_log;  // 平均后处理耗时。
            double avg_infer = infer_sum_since_log / rendered_since_log;  // 平均整体推理耗时。
            double avg_render_queue = render_queue_sum_since_log / rendered_since_log;  // 平均推理后等待渲染耗时。
            double avg_latency = latency_sum_since_log / rendered_since_log;  // 平均端到端延迟。
            PrintRuntimeStats(runtime_->stats,
                              rendered_since_log,
                              elapsed_since_log,
                              avg_decode,
                              avg_decode_queue,
                              avg_preprocess,
                              avg_npu,
                              avg_postprocess,
                              avg_infer,
                              avg_render_queue,
                              avg_latency);

            rendered_since_log = 0;
            decode_sum_since_log = 0.0;
            decode_queue_sum_since_log = 0.0;
            infer_sum_since_log = 0.0;
            preprocess_sum_since_log = 0.0;
            npu_sum_since_log = 0.0;
            postprocess_sum_since_log = 0.0;
            render_queue_sum_since_log = 0.0;
            latency_sum_since_log = 0.0;
            last_log_ms = now;
        }
    }

    // 线程结束前，如果还有没输出的统计窗口，补打一份。
    if (rendered_since_log > 0) {
        double elapsed_since_log = NowMs() - last_log_ms;  // 尾窗口实际持续时间。
        double avg_decode = decode_sum_since_log / rendered_since_log;  // 平均解码耗时。
        double avg_decode_queue = decode_queue_sum_since_log / rendered_since_log;  // 平均解码后等待推理耗时。
        double avg_preprocess = preprocess_sum_since_log / rendered_since_log;  // 平均预处理耗时。
        double avg_npu = npu_sum_since_log / rendered_since_log;  // 平均 NPU 耗时。
        double avg_postprocess = postprocess_sum_since_log / rendered_since_log;  // 平均后处理耗时。
        double avg_infer = infer_sum_since_log / rendered_since_log;  // 平均整体推理耗时。
        double avg_render_queue = render_queue_sum_since_log / rendered_since_log;  // 平均推理后等待渲染耗时。
        double avg_latency = latency_sum_since_log / rendered_since_log;  // 平均端到端延迟。
        PrintRuntimeStats(runtime_->stats,
                          rendered_since_log,
                          elapsed_since_log,
                          avg_decode,
                          avg_decode_queue,
                          avg_preprocess,
                          avg_npu,
                          avg_postprocess,
                          avg_infer,
                          avg_render_queue,
                          avg_latency);
    }

    drm_display.Close();
}

}  // namespace rknn_demo
