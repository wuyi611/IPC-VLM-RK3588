#ifndef RKNN_YOLOV5_DEMO_PIPELINE_PIPELINE_TYPES_H_
#define RKNN_YOLOV5_DEMO_PIPELINE_PIPELINE_TYPES_H_

#include <stdint.h>

#include <atomic>
#include <string.h>

#include <rockchip/mpp_frame.h>

#include "media/ffmpeg_headers.h"
#include "model/postprocess.h"

// 文件说明：
// 定义流水线阶段之间传递的数据包、运行时状态和统计信息结构。

namespace rknn_demo {

// 表示采集阶段输出的一份压缩视频包。
// 这个结构把 FFmpeg 的 `AVPacket` 连同项目自己的时间戳和帧编号一起传递。
struct EncodedPacket {
    // 本项目内部生成的顺序帧编号。
    uint64_t frame_id;

    // 该帧被采集线程读到时的时间戳，单位毫秒。
    double capture_ts_ms;

    // 是否为流结束标记。
    bool end_of_stream;

    // FFmpeg 原始压缩包数据。
    AVPacket packet;

    // 默认构造函数，负责把 `AVPacket` 初始化为空包。
    EncodedPacket()
        : frame_id(0),
          capture_ts_ms(0.0),
          end_of_stream(false) {
        memset(&packet, 0, sizeof(packet));
        packet.pts = AV_NOPTS_VALUE;
        packet.dts = AV_NOPTS_VALUE;
        packet.pos = -1;
    }

    // 析构时释放 `AVPacket` 内部资源。
    ~EncodedPacket() {
        av_packet_unref(&packet);
    }

    // 禁止拷贝，避免 `AVPacket` 发生隐式深浅拷贝问题。
    EncodedPacket(const EncodedPacket &) = delete;
    EncodedPacket &operator=(const EncodedPacket &) = delete;

    // 允许移动构造，把 `AVPacket` 的所有权安全转移给新对象。
    EncodedPacket(EncodedPacket &&other) noexcept
        : frame_id(other.frame_id),
          capture_ts_ms(other.capture_ts_ms),
          end_of_stream(other.end_of_stream) {
        memset(&packet, 0, sizeof(packet));
        packet.pts = AV_NOPTS_VALUE;
        packet.dts = AV_NOPTS_VALUE;
        packet.pos = -1;
        av_packet_move_ref(&packet, &other.packet);
        other.frame_id = 0;
        other.capture_ts_ms = 0.0;
        other.end_of_stream = false;
    }

    // 允许移动赋值，用于跨线程队列中转移包数据所有权。
    EncodedPacket &operator=(EncodedPacket &&other) noexcept {
        if (this != &other) {
            av_packet_unref(&packet);
            frame_id = other.frame_id;
            capture_ts_ms = other.capture_ts_ms;
            end_of_stream = other.end_of_stream;
            av_packet_move_ref(&packet, &other.packet);
            other.frame_id = 0;
            other.capture_ts_ms = 0.0;
            other.end_of_stream = false;
        }
        return *this;
    }
};

// 对 `MppFrame` 做轻量 RAII 封装。
// 这个句柄用于在“纯 MPP 解码 -> RGA -> RKNN”链路中跨线程传递 DMA-BUF 帧。
struct MppFrameHandle {
    // 实际持有的 MPP 帧句柄。
    MppFrame frame;

    // 供后续 RGA 直接消费的图像描述。
    image_buffer_t image;

    // 构造一个空句柄。
    MppFrameHandle()
        : frame(NULL) {
        memset(&image, 0, sizeof(image));
        image.fd = -1;
    }

    // 析构时自动释放持有的 MPP 帧。
    ~MppFrameHandle() {
        Reset();
    }

    // 禁止拷贝，避免同一帧句柄被重复释放。
    MppFrameHandle(const MppFrameHandle &) = delete;
    MppFrameHandle &operator=(const MppFrameHandle &) = delete;

    // 允许移动构造，把帧所有权转移给新对象。
    MppFrameHandle(MppFrameHandle &&other) noexcept
        : frame(other.frame),
          image(other.image) {
        other.frame = NULL;
        memset(&other.image, 0, sizeof(other.image));
        other.image.fd = -1;
    }

    // 允许移动赋值，便于结果包在队列间转移。
    MppFrameHandle &operator=(MppFrameHandle &&other) noexcept {
        if (this != &other) {
            Reset();
            frame = other.frame;
            image = other.image;
            other.frame = NULL;
            memset(&other.image, 0, sizeof(other.image));
            other.image.fd = -1;
        }
        return *this;
    }

    // 重置当前句柄，并可选接管一帧新的 MPP 帧和图像描述。
    void Reset(MppFrame new_frame = NULL, const image_buffer_t *new_image = NULL) {
        if (frame != NULL) {
            mpp_frame_deinit(&frame);
        }
        frame = new_frame;
        memset(&image, 0, sizeof(image));
        image.fd = -1;
        if (new_image != NULL) {
            image = *new_image;
        }
    }

    // 判断当前是否未持有任何有效帧。
    bool empty() const {
        return frame == NULL;
    }
};

// 表示解码阶段输出的一帧数据。
struct DecodedPacket {
    // 与输入压缩包对应的帧编号。
    uint64_t frame_id;

    // 该帧进入系统时的采集时间戳。
    double capture_ts_ms;

    // 解码完成时间戳，单位毫秒。
    double decode_done_ts_ms;

    // 实际解码得到的 MPP DMA-BUF 帧。
    MppFrameHandle decoded_frame;
};

// 表示推理阶段输出的一份结果。
// 这里除了检测框，还保留各阶段时间信息，方便渲染阶段做统计。
struct ResultPacket {
    // 帧编号。
    uint64_t frame_id;

    // 采集时间戳。
    double capture_ts_ms;

    // 解码完成时间戳。
    double decode_done_ts_ms;

    // 推理开始时间戳。
    double infer_start_ts_ms;

    // 推理完成时间戳。
    double infer_done_ts_ms;

    // 整体推理耗时。
    double infer_ms;

    // 预处理耗时。
    double preprocess_ms;

    // NPU 执行耗时。
    double npu_ms;

    // 后处理耗时。
    double postprocess_ms;

    // 用于显示的帧。
    // 纯 MPP DMA-BUF 渲染路径下，这里保存原始解码帧句柄。
    MppFrameHandle display_frame;

    // 当前帧的目标检测结果列表。
    object_detect_result_list detections;
};

// 运行时统计信息。
// 所有字段都用原子变量，便于多线程并发更新而不额外加锁。
struct RuntimeStats {
    // 成功采集的帧数。
    std::atomic<uint64_t> captured;

    // 成功解码的帧数。
    std::atomic<uint64_t> decoded;

    // 成功完成推理的帧数。
    std::atomic<uint64_t> inferred;

    // 成功被渲染线程消费的帧数。
    std::atomic<uint64_t> rendered;

    // 因采集队列满而丢弃的帧数。
    std::atomic<uint64_t> dropped_capture_queue;

    // 因解码队列满而丢弃的帧数。
    std::atomic<uint64_t> dropped_decode_queue;

    // 因渲染队列满而丢弃的帧数。
    std::atomic<uint64_t> dropped_render_queue;

    // 构造时把所有统计值清零。
    RuntimeStats()
        : captured(0),
          decoded(0),
          inferred(0),
          rendered(0),
          dropped_capture_queue(0),
          dropped_decode_queue(0),
          dropped_render_queue(0) {}
};

// 整条流水线共享的运行时状态。
struct PipelineRuntime {
    // 是否已经请求全局停止。
    std::atomic<bool> stop_requested;

    // 当前仍在运行的推理 worker 数量。
    std::atomic<int> active_infer_workers;

    // 全局统计信息。
    RuntimeStats stats;

    // 根据推理线程数初始化运行时状态。
    explicit PipelineRuntime(int infer_thread_count)
        : stop_requested(false),
          active_infer_workers(infer_thread_count) {}
};

}  // namespace rknn_demo

#endif  // RKNN_YOLOV5_DEMO_PIPELINE_PIPELINE_TYPES_H_
