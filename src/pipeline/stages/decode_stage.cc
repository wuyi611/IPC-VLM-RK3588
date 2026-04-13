#include "pipeline/stages/decode_stage.h"

#include <stdio.h>
#include <thread>

#include "media/ffmpeg_headers.h"
#include "pipeline/pipeline_utils.h"

// 文件说明：
// 实现解码线程逻辑，并把解码后的 DMA-BUF 视频帧送入推理前队列。

namespace rknn_demo {

// 保存解码阶段运行所需的流信息、队列和共享状态。
DecodeStage::DecodeStage(const MediaStreamInfo &stream_info,
                         PipelineQueues *queues,
                         PipelineRuntime *runtime)
    : stream_info_(stream_info),
      initialized_(false),
      queues_(queues),
      runtime_(runtime) {}

// 用探测阶段拿到的流参数初始化底层纯 MPP 解码器。
bool DecodeStage::Init() {
    initialized_ = decoder_.Init(stream_info_);
    if (initialized_) {
        printf("decode: initialized pure mpp codec=%s size=%dx%d\n",
               avcodec_get_name(stream_info_.codec_id),
               stream_info_.width,
               stream_info_.height);
    }
    return initialized_;
}

// 持续消费压缩包，并把成功解出的帧送往推理阶段。
void DecodeStage::Run() {
    if (!initialized_) {
        printf("Error: pure mpp decoder was not initialized before decode thread start.\n");
        queues_->decode_queue().stop();
        return;
    }

    const std::chrono::milliseconds kRetryDelay(1);  // 解码器暂时繁忙时的短暂退避时间。
    bool first_frame_logged = false;  // 控制首帧日志只打印一次。

    // 尽可能把当前已经解好的帧全部取出来并送往下游。
    auto DrainFrames = [this, &first_frame_logged]() -> bool {
        bool got_frame = false;  // 记录这次 drain 是否真的拿到了输出帧。
        while (true) {
            DecodedPacket decoded;  // 当前从解码器取出的输出帧包。
            if (!decoder_.ReceiveFrame(&decoded)) {
                break;
            }
            decoded.decode_done_ts_ms = NowMs();

            if (!first_frame_logged && !decoded.decoded_frame.empty()) {
                const image_buffer_t &image = decoded.decoded_frame.image;  // 首帧对应的图像描述。
                printf("decode: first dma-buf frame ready, size=%dx%d stride=%dx%d fd=%d\n",
                       image.width,
                       image.height,
                       image.width_stride,
                       image.height_stride,
                       image.fd);
                first_frame_logged = true;
            }

            bool dropped = false;  // 记录本次写入是否因为队列已满而丢掉了旧帧。
            if (!queues_->decode_queue().push(std::move(decoded), &dropped)) {
                queues_->decode_queue().stop();
                return false;
            }

            runtime_->stats.decoded.fetch_add(1, std::memory_order_relaxed);
            if (dropped) {
                runtime_->stats.dropped_decode_queue.fetch_add(1, std::memory_order_relaxed);
            }
            got_frame = true;
        }
        return got_frame;
    };

    EncodedPacket encoded;  // 当前从采集队列取出的压缩包。
    while (!runtime_->stop_requested.load() &&
           !decoder_.HasFatalError() &&
           queues_->capture_queue().pop(&encoded)) {
        if (encoded.end_of_stream) {
            // 收到 EOS 后，先把 EOS 送进解码器，再持续排空内部缓冲。
            while (!runtime_->stop_requested.load()) {
                MppDecoder::FeedStatus status = decoder_.SendEos();  // 当前 EOS 投递状态。
                if (status == MppDecoder::kFeedOk) {
                    break;
                }
                if (status == MppDecoder::kFeedError) {
                    queues_->decode_queue().stop();
                    return;
                }
                if (!DrainFrames()) {
                    std::this_thread::sleep_for(kRetryDelay);
                }
            }

            while (!runtime_->stop_requested.load()) {
                bool got_frame = DrainFrames();  // 当前这轮是否成功取出了输出帧。
                if (decoder_.Drained()) {
                    break;
                }
                if (!got_frame) {
                    std::this_thread::sleep_for(kRetryDelay);
                }
            }
            if (decoder_.HasFatalError()) {
                break;
            }
        } else {
            // 普通压缩包走“送包 -> 尝试取帧”的循环。
            while (!runtime_->stop_requested.load()) {
                MppDecoder::FeedStatus status = decoder_.SendPacket(encoded);  // 当前送包状态。
                if (status == MppDecoder::kFeedOk) {
                    break;
                }
                if (status == MppDecoder::kFeedError) {
                    queues_->decode_queue().stop();
                    return;
                }
                if (!DrainFrames()) {
                    std::this_thread::sleep_for(kRetryDelay);
                }
            }
            DrainFrames();
        }

        if (encoded.end_of_stream) {
            break;
        }
    }

    // 解码阶段结束后关闭 decode 队列，通知推理线程后续不会再有新帧。
    queues_->decode_queue().stop();
}

}  // namespace rknn_demo
