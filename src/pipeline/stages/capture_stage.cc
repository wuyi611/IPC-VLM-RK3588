#include "pipeline/stages/capture_stage.h"

#include <stdio.h>

#include "pipeline/pipeline_utils.h"

// 文件说明：
// 实现采集线程逻辑，并在结束时向后续阶段发送 EOS 信号。

namespace rknn_demo {

// 保存采集阶段所需的共享对象和输入源句柄。
CaptureStage::CaptureStage(const AppOptions &options,
                           std::unique_ptr<FfmpegDemuxer> demuxer,
                           PipelineQueues *queues,
                           PipelineRuntime *runtime)
    : options_(options),
      demuxer_(std::move(demuxer)),
      queues_(queues),
      runtime_(runtime) {}

// 持续从 demuxer 读取压缩包，并把 EOS 统一转换成队列里的结束标记。
void CaptureStage::Run() {
    (void)options_;

    if (!demuxer_) {
        printf("Error: FFmpeg demuxer is not initialized.\n");
        RequestPipelineStop(runtime_, queues_);
        return;
    }

    uint64_t next_frame_id = 0;  // 本地递增的帧编号计数器。
    while (!runtime_->stop_requested.load()) {
        EncodedPacket packet;  // 当前要送入队列的一份压缩包。
        if (!demuxer_->ReadPacket(&packet.packet)) {
            break;
        }

        packet.frame_id = next_frame_id++;
        packet.capture_ts_ms = NowMs();
        packet.end_of_stream = false;

        bool dropped = false;  // 记录本次写入是否因为队列已满而丢掉了旧帧。
        if (!queues_->capture_queue().push(std::move(packet), &dropped)) {
            break;
        }

        runtime_->stats.captured.fetch_add(1, std::memory_order_relaxed);
        if (dropped) {
            runtime_->stats.dropped_capture_queue.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // 正常结束时补一个 EOS 包，通知解码阶段可以开始 flush。
    // 这个包本身不携带真实码流数据，只承担“输入结束”的信号作用。
    EncodedPacket eos_packet;
    eos_packet.frame_id = next_frame_id;
    eos_packet.capture_ts_ms = NowMs();
    eos_packet.end_of_stream = true;

    // 不关心 EOS 是否挤掉旧包，因为这里只需要尽量把结束信号送达。
    bool dropped = false;
    queues_->capture_queue().push(std::move(eos_packet), &dropped);

    // 关闭采集队列，唤醒后续可能阻塞等待的消费者。
    queues_->capture_queue().stop();
}

}  // namespace rknn_demo
