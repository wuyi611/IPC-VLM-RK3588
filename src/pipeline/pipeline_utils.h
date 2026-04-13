#ifndef RKNN_YOLOV5_DEMO_PIPELINE_PIPELINE_UTILS_H_
#define RKNN_YOLOV5_DEMO_PIPELINE_PIPELINE_UTILS_H_

#include <sys/time.h>

#include "app/app_options.h"
#include "common.h"
#include "pipeline/pipeline_queues.h"
#include "pipeline/pipeline_types.h"

// 文件说明：
// 声明流水线运行中会复用的一组时间、日志和停机辅助函数。

namespace rknn_demo {

// 把 `timeval` 转换成微秒数表示。
double GetUs(const struct timeval &time_value);

// 获取当前单调时钟时间，单位毫秒。
double NowMs();

// 根据 worker 序号和推理线程数选择目标 NPU 核心掩码。
rknn_core_mask SelectCoreMask(int worker_index, int infer_thread_count);

// 根据配置决定是否打印当前帧的检测结果。
void LogDetectionsIfNeeded(const AppOptions &options, const ResultPacket &result);

// 输出一段时间窗口内的运行统计信息。
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
                       double avg_latency_ms);

// 请求整条流水线停止运行。
// 这个辅助函数会统一设置停止标志，并停止所有队列。
void RequestPipelineStop(PipelineRuntime *runtime, PipelineQueues *queues);

}  // namespace rknn_demo

#endif  // RKNN_YOLOV5_DEMO_PIPELINE_PIPELINE_UTILS_H_
