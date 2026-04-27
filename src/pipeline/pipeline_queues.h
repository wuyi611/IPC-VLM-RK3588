#ifndef RKNN_YOLOV5_DEMO_PIPELINE_PIPELINE_QUEUES_H_
#define RKNN_YOLOV5_DEMO_PIPELINE_PIPELINE_QUEUES_H_

#include <cstddef>

#include "pipeline/bounded_queue.h"
#include "pipeline/pipeline_types.h"

// 文件说明：
// 集中管理整条流水线中的三段核心队列，方便各阶段统一访问和统一停机。

namespace rknn_demo {

// `PipelineQueues` 是一个轻量容器，
// 把采集、解码、渲染相关的跨线程队列集中到一起管理。
class PipelineQueues {
public:
    // 使用三段队列各自的容量创建队列集合。
    PipelineQueues(size_t capture_queue_size,
                   size_t decode_queue_size,
                   size_t render_queue_size,
                   size_t llm_queue_size)
        : capture_queue_(capture_queue_size),
          decode_queue_(decode_queue_size),
          render_queue_(render_queue_size),
          llm_queue_(llm_queue_size) {}

    // 返回“采集 -> 解码”阶段之间的压缩包队列。
    BoundedQueue<EncodedPacket> &capture_queue() { return capture_queue_; }

    // 返回“解码 -> 推理”阶段之间的解码帧队列。
    BoundedQueue<DecodedPacket> &decode_queue() { return decode_queue_; }

    // 返回“推理 -> 渲染”阶段之间的结果队列。
    BoundedQueue<ResultPacket> &render_queue() { return render_queue_; }

    // 返回“推理 -> LLM”阶段之间的抽样请求队列。
    BoundedQueue<LlmRequestPacket> &llm_queue() { return llm_queue_; }

    // 停止整条流水线的全部队列。
    // 一旦某个阶段判断程序需要退出，就可以调用这里唤醒所有等待线程。
    void StopAll() {
        capture_queue_.stop();
        decode_queue_.stop();
        render_queue_.stop();
        llm_queue_.stop();
    }

private:
    // 存放压缩视频包的队列。
    BoundedQueue<EncodedPacket> capture_queue_;

    // 存放解码后 MPP 帧的队列。
    BoundedQueue<DecodedPacket> decode_queue_;

    // 存放推理结果和显示帧的队列。
    BoundedQueue<ResultPacket> render_queue_;

    // 存放送往 LLM 的抽样请求队列。
    BoundedQueue<LlmRequestPacket> llm_queue_;
};

}  // namespace rknn_demo

#endif  // RKNN_YOLOV5_DEMO_PIPELINE_PIPELINE_QUEUES_H_
