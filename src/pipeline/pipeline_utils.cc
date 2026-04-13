#include "pipeline/pipeline_utils.h"

#include <stdio.h>

#include <chrono>

// 文件说明：
// 实现流水线公共辅助函数，包括计时、日志输出和统一停机入口。

namespace rknn_demo {

// 把 `timeval` 结构拍平成一个便于做差值计算的微秒整数。
double GetUs(const struct timeval &time_value) {
    return (time_value.tv_sec * 1000000.0 + time_value.tv_usec);
}

// 基于单调时钟返回当前毫秒值，避免受系统时间回拨影响。
double NowMs() {
    using namespace std::chrono;
    return duration_cast<duration<double, std::milli> >(
               steady_clock::now().time_since_epoch())
        .count();
}

// 根据 worker 编号挑选 NPU 核心，尽量让多线程时的负载更均匀。
rknn_core_mask SelectCoreMask(int worker_index, int infer_thread_count) {
#if defined(__aarch64__)
    // 当只有一个推理线程时，直接交给全部 NPU 核心联合调度。
    if (infer_thread_count <= 1) {
        return RKNN_NPU_CORE_0_1_2;
    }

    // 多线程时按 worker 序号轮流分配到三个 NPU 核心上。
    switch (worker_index % 3) {
    case 0:
        return RKNN_NPU_CORE_0;
    case 1:
        return RKNN_NPU_CORE_1;
    default:
        return RKNN_NPU_CORE_2;
    }
#else
    (void)worker_index;
    (void)infer_thread_count;
    return RKNN_NPU_CORE_AUTO;
#endif
}

// 在开启检测日志时，把当前结果包里的每个检测对象逐行输出。
void LogDetectionsIfNeeded(const AppOptions &options, const ResultPacket &result) {
    if (!options.log_detections) {
        return;
    }

    // 逐个打印当前帧中的检测对象。
    for (int i = 0; i < result.detections.count; ++i) {
        const object_detect_result &det = result.detections.results[i];
        printf("frame=%llu %s @ (%d %d %d %d) %.3f\n",
               static_cast<unsigned long long>(result.frame_id),
               coco_cls_to_name(det.cls_id),
               det.box.left,
               det.box.top,
               det.box.right,
               det.box.bottom,
               det.prop);
    }
}

// 打印一个统计窗口内聚合得到的吞吐、延迟和丢帧指标。
void PrintRuntimeStats(const RuntimeStats &stats,
                       uint64_t interval_frames,
                       double interval_ms,
                       double avg_decode_ms,
                       double avg_decode_queue_ms,
                       double avg_preprocess_ms,
                       double avg_npu_ms,
                       double avg_postprocess_ms,
                       double avg_infer_ms,
                       double avg_render_queue_ms,
                       double avg_latency_ms) {
    // 当前统计窗口内的平均渲染 FPS。
    double fps = interval_ms > 0.0 ? (interval_frames * 1000.0 / interval_ms) : 0.0;
    printf("stats: fps=%.2f decode=%.2fms wait_infer=%.2fms pre=%.2fms npu=%.2fms post=%.2fms infer=%.2fms wait_render=%.2fms latency=%.2fms captured=%llu decoded=%llu inferred=%llu rendered=%llu drop_cap=%llu drop_dec=%llu drop_rnd=%llu\n",
           fps,
           avg_decode_ms,
           avg_decode_queue_ms,
           avg_preprocess_ms,
           avg_npu_ms,
           avg_postprocess_ms,
           avg_infer_ms,
           avg_render_queue_ms,
           avg_latency_ms,
           static_cast<unsigned long long>(stats.captured.load(std::memory_order_relaxed)),
           static_cast<unsigned long long>(stats.decoded.load(std::memory_order_relaxed)),
           static_cast<unsigned long long>(stats.inferred.load(std::memory_order_relaxed)),
           static_cast<unsigned long long>(stats.rendered.load(std::memory_order_relaxed)),
           static_cast<unsigned long long>(stats.dropped_capture_queue.load(std::memory_order_relaxed)),
           static_cast<unsigned long long>(stats.dropped_decode_queue.load(std::memory_order_relaxed)),
           static_cast<unsigned long long>(stats.dropped_render_queue.load(std::memory_order_relaxed)));
}

// 向所有阶段广播停机信号，只让第一次调用真正生效。
void RequestPipelineStop(PipelineRuntime *runtime, PipelineQueues *queues) {
    if (runtime == NULL || queues == NULL) {
        return;
    }

    // 只允许第一次请求真正触发全局停机，
    // 避免多个线程重复调用时产生多余副作用。
    bool expected = false;  // CAS 前期望看到的旧值。
    if (runtime->stop_requested.compare_exchange_strong(expected, true)) {
        queues->StopAll();
    }
}

}  // namespace rknn_demo
