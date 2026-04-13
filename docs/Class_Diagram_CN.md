# IPCYolo-rk3588 类图

下面的类图聚焦当前项目的核心 C++ 类与关键结构体。

说明：
- 图中同时包含 `class` 和关键 `struct`，因为这个项目大量通过结构体在模块间传递数据。
- `main()`、`init_yolov5_model()`、`post_process()` 这类自由函数没有画成类。
- Mermaid 可在大多数 Markdown 预览器中直接渲染。

## 核心编排类图

```mermaid
classDiagram
direction LR

class AppOptions {
  +string model_path
  +string input
  +string model_type
  +string labels_path
  +int infer_thread_count
  +int capture_queue_size
  +int decode_queue_size
  +int render_queue_size
  +int stats_interval
  +bool enable_display
  +bool log_detections
}

class AppOptionsParser {
  +Parse(argc, argv, options)$ bool
  +PrintUsage(program)$ void
}

class PipelineApp {
  -AppOptions options_
  -PipelineQueues queues_
  -PipelineRuntime runtime_
  -infer_contexts_: multiple rknn_app_context_t
  +PipelineApp(options)
  +Run() int
}

class MediaStreamInfo {
  +MppCodingType coding_type
  +int width
  +int height
  +AVRational time_base
  +extra_data: byte array
}

class FfmpegDemuxer {
  -AVFormatContext* format_context_
  -int video_stream_index_
  +Open(input, stream_info) bool
  +ReadPacket(packet) bool
  +Close() void
}

class MppDecoder {
  -MediaStreamInfo stream_info_
  -bool initialized_
  -bool extra_data_sent_
  +Init(stream_info) bool
  +SendPacket(packet, decoded_packet) bool
  +SendEos(decoded_packet) bool
  +Reset() void
}

class CaptureStage {
  -AppOptions& options_
  -demuxer_: FfmpegDemuxer owner
  -PipelineQueues* queues_
  -PipelineRuntime* runtime_
  +Run() void
}

class DecodeStage {
  -MediaStreamInfo stream_info_
  -MppDecoder decoder_
  -bool initialized_
  -PipelineQueues* queues_
  -PipelineRuntime* runtime_
  +Init() bool
  +Run() void
}

class InferenceStage {
  -int worker_id_
  -AppOptions& options_
  -rknn_app_context_t* app_context_
  -PipelineQueues* queues_
  -PipelineRuntime* runtime_
  +Run() void
}

class RenderStage {
  -AppOptions& options_
  -PipelineQueues* queues_
  -PipelineRuntime* runtime_
  +Run() void
}

class PipelineQueues {
  -capture_queue_
  -decode_queue_
  -render_queue_
  +capture_queue()
  +decode_queue()
  +render_queue()
  +StopAll() void
}

class PipelineRuntime {
  +atomic<bool> stop_requested
  +atomic<int> active_infer_workers
  +RuntimeStats stats
}

class RuntimeStats {
  +atomic<uint64_t> captured
  +atomic<uint64_t> decoded
  +atomic<uint64_t> inferred
  +atomic<uint64_t> rendered
  +atomic<uint64_t> dropped_capture_queue
  +atomic<uint64_t> dropped_decode_queue
  +atomic<uint64_t> dropped_render_queue
}

class BoundedQueue~T~ {
  +push(item, dropped_oldest) bool
  +pop(item) bool
  +stop() void
}

class BoundedQueue_EncodedPacket
class BoundedQueue_DecodedPacket
class BoundedQueue_ResultPacket

class rknn_app_context_t {
  +rknn_context rknn_ctx
  +rknn_input_output_num io_num
  +rknn_tensor_attr* input_attrs
  +rknn_tensor_attr* output_attrs
  +int model_channel
  +int model_width
  +int model_height
  +bool is_quant
  +image_buffer_t model_input_buffer
}

AppOptionsParser ..> AppOptions : fills
PipelineApp *-- AppOptions
PipelineApp *-- PipelineQueues
PipelineApp *-- PipelineRuntime
PipelineApp *-- "1..*" rknn_app_context_t

CaptureStage --> AppOptions
CaptureStage *-- FfmpegDemuxer
CaptureStage --> PipelineQueues
CaptureStage --> PipelineRuntime

DecodeStage *-- MppDecoder
DecodeStage *-- MediaStreamInfo
DecodeStage --> PipelineQueues
DecodeStage --> PipelineRuntime

InferenceStage --> AppOptions
InferenceStage --> rknn_app_context_t
InferenceStage --> PipelineQueues
InferenceStage --> PipelineRuntime

RenderStage --> AppOptions
RenderStage --> PipelineQueues
RenderStage --> PipelineRuntime

FfmpegDemuxer ..> MediaStreamInfo : outputs
MppDecoder *-- MediaStreamInfo
PipelineRuntime *-- RuntimeStats
PipelineQueues *-- BoundedQueue_EncodedPacket
PipelineQueues *-- BoundedQueue_DecodedPacket
PipelineQueues *-- BoundedQueue_ResultPacket
BoundedQueue_EncodedPacket ..> BoundedQueue~T~
BoundedQueue_DecodedPacket ..> BoundedQueue~T~
BoundedQueue_ResultPacket ..> BoundedQueue~T~
```

## 数据结构关系图

```mermaid
classDiagram
direction LR

class EncodedPacket {
  +uint64_t frame_id
  +double capture_ts_ms
  +bool end_of_stream
  +AVPacket packet
}

class MppFrameHandle {
  +MppFrame frame
  +Reset(new_frame) void
  +empty() bool
}

class DecodedPacket {
  +uint64_t frame_id
  +double capture_ts_ms
  +double decode_done_ts_ms
  +MppFrameHandle mpp_frame
}

class object_detect_result {
  +image_rect_t box
  +float prop
  +int cls_id
}

class object_detect_result_list {
  +int id
  +int count
  +object_detect_result results[OBJ_NUMB_MAX_SIZE]
}

class ResultPacket {
  +uint64_t frame_id
  +double capture_ts_ms
  +double decode_done_ts_ms
  +double infer_start_ts_ms
  +double infer_done_ts_ms
  +double infer_ms
  +double preprocess_ms
  +double npu_ms
  +double postprocess_ms
  +MppFrameHandle display_frame
  +object_detect_result_list detections
}

class InferenceProfile {
  +double preprocess_ms
  +double npu_ms
  +double postprocess_ms
  +double total_ms
}

EncodedPacket *-- AVPacket
DecodedPacket *-- MppFrameHandle
ResultPacket *-- MppFrameHandle
ResultPacket *-- object_detect_result_list
object_detect_result_list *-- object_detect_result
```

## 额外说明

- `PipelineApp` 是总编排器，负责创建并启动 `CaptureStage -> DecodeStage -> InferenceStage -> RenderStage`。
- `PipelineQueues` 是跨阶段共享的数据通道，底层基于 `BoundedQueue<T>`。
- `rknn_app_context_t` 不是类，但它是推理阶段最核心的模型上下文，因此在类图中单独展示。
- YOLOv5 / YOLOv8 的模型加载、推理和后处理当前主要通过自由函数围绕 `rknn_app_context_t` 工作，而不是面向对象类封装。
