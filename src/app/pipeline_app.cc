#include "app/pipeline_app.h"

#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include <memory>
#include <atomic>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "media/ffmpeg_demuxer.h"
#include "model/postprocess.h"
#include "model/yolov5_model.h"
#include "model/yolov8_model.h"
#include "pipeline/pipeline_utils.h"
#include "pipeline/stages/capture_stage.h"
#include "pipeline/stages/decode_stage.h"
#include "pipeline/stages/inference_stage.h"
#include "pipeline/stages/render_stage.h"

// 文件说明：
// 实现 `PipelineApp` 的核心编排逻辑，包括模型初始化、
// 流信息探测、各阶段对象创建、线程启动和最终资源释放。

namespace rknn_demo {

namespace {

// 判断当前配置是否使用 YOLOv8 模型
bool UseYolov8Model(const AppOptions &options) {
    return options.model_type == "yolov8";
}

// 根据信号编号返回对应的信号名称字符串，用于日志打印
const char *SignalName(int signal_number) {
    switch (signal_number) {
    case SIGINT:
        return "SIGINT";
    case SIGTERM:
        return "SIGTERM";
    case SIGHUP:
        return "SIGHUP";
    default:
        return "UNKNOWN";
    }
}

// `SignalWatcher` 把进程信号转换成流水线内部的停止请求，
// 避免在异步信号处理函数里直接做复杂清理。
class SignalWatcher {
public:
    // SignalWatcher 构造函数，初始化需要监听的信号集（SIGINT, SIGTERM, SIGHUP）和内部状态
    SignalWatcher(PipelineRuntime *runtime, PipelineQueues *queues)
        : runtime_(runtime),
          queues_(queues),
          thread_stop_requested_(false),
          received_signal_(0),
          started_(false),
          mask_installed_(false) {
        sigemptyset(&signal_mask_);
        sigaddset(&signal_mask_, SIGINT);
        sigaddset(&signal_mask_, SIGTERM);
        sigaddset(&signal_mask_, SIGHUP);
        memset(&old_mask_, 0, sizeof(old_mask_));
    }

    // SignalWatcher 析构函数，确保在销毁时停止监听线程并恢复信号掩码
    ~SignalWatcher() {
        Stop();
    }

    // 启动信号监听。会阻塞当前线程的特定信号，并创建一个后台线程通过 sigtimedwait 专门处理这些信号
    bool Start() {
        if (started_) {
            return true;
        }

        int ret = pthread_sigmask(SIG_BLOCK, &signal_mask_, &old_mask_);  // 先在当前线程安装信号屏蔽字。
        if (ret != 0) {
            printf("signal: pthread_sigmask failed ret=%d\n", ret);
            return false;
        }
        mask_installed_ = true;

        watcher_thread_ = std::thread(&SignalWatcher::Run, this);  // 后台线程专门负责等待退出信号。
        started_ = true;
        return true;
    }

    // 停止信号监听线程，并恢复原有的系统信号屏蔽字
    void Stop() {
        if (!started_) {
            return;
        }

        thread_stop_requested_.store(true, std::memory_order_relaxed);
        if (watcher_thread_.joinable()) {
            watcher_thread_.join();
        }

        if (mask_installed_) {
            pthread_sigmask(SIG_SETMASK, &old_mask_, NULL);
            mask_installed_ = false;
        }
        started_ = false;
    }

    // 检查是否接收到了系统中断/退出信号
    bool interrupted() const {
        return received_signal_.load(std::memory_order_relaxed) != 0;
    }

    // 获取实际接收到的具体信号编号
    int received_signal() const {
        return received_signal_.load(std::memory_order_relaxed);
    }

    // 根据接收到的信号计算程序的标准退出状态码 (通常为 128 + 信号编号)
    int ExitCode() const {
        int signal_number = received_signal();
        return signal_number > 0 ? 128 + signal_number : 0;
    }

private:
    // 信号监听线程的实际运行循环，持续等待信号并在触发时请求关闭整个流水线
    void Run() {
        while (!thread_stop_requested_.load(std::memory_order_relaxed)) {
            struct timespec timeout;  // 轮询等待超时，避免线程永久阻塞在 `sigtimedwait`。
            timeout.tv_sec = 0;
            timeout.tv_nsec = 200 * 1000 * 1000;

            siginfo_t signal_info;  // 接收系统返回的信号附带信息。
            memset(&signal_info, 0, sizeof(signal_info));
            int signal_number = sigtimedwait(&signal_mask_, &signal_info, &timeout);  // 实际收到的信号编号。
            if (signal_number < 0) {
                if (errno == EAGAIN || errno == EINTR) {
                    continue;
                }
                printf("signal: sigtimedwait failed errno=%d\n", errno);
                break;
            }

            int expected = 0;  // 只有首次收到信号时才打印“开始优雅退出”日志。
            if (received_signal_.compare_exchange_strong(expected, signal_number)) {
                printf("signal: received %s, requesting graceful shutdown\n",
                       SignalName(signal_number));
            } else {
                printf("signal: received %s again, shutdown already in progress\n",
                       SignalName(signal_number));
            }
            RequestPipelineStop(runtime_, queues_);
        }
    }

    PipelineRuntime *runtime_;  // 全局运行时状态，停止请求最终会写入这里。
    PipelineQueues *queues_;  // 共享队列集合，停止时要顺带唤醒可能阻塞的阶段。
    std::atomic<bool> thread_stop_requested_;  // 控制监听线程自行退出的内部标记。
    std::atomic<int> received_signal_;  // 记录首个触发优雅停机的系统信号编号。
    std::thread watcher_thread_;  // 专门等待系统信号的后台线程。
    sigset_t signal_mask_;  // 需要由监听线程统一接管的信号集合。
    sigset_t old_mask_;  // 进入监听前原有的线程信号屏蔽字。
    bool started_;  // 标记监听线程是否已经启动成功。
    bool mask_installed_;  // 标记当前线程是否已经安装了新的信号屏蔽字。
};

}  // namespace

// PipelineApp 构造函数：初始化应用程序选项、各个阶段的缓冲队列大小，以及为每个推理线程分配上下文槽位
PipelineApp::PipelineApp(const AppOptions &options)
    : options_(options),
      // 队列深度由启动参数决定，构造时直接固定下来。
      queues_(static_cast<size_t>(options.capture_queue_size),
              static_cast<size_t>(options.decode_queue_size),
              static_cast<size_t>(options.render_queue_size)),
      // 运行时状态需要知道推理线程数，用于统计和退出协调。
      runtime_(options.infer_thread_count),
      // 为每个推理线程预留一个独立的 RKNN 上下文槽位。
      infer_contexts_(options.infer_thread_count) {}

// 初始化所有推理工作线程的模型上下文，包括模型加载及底层 NPU 核心绑定
bool PipelineApp::InitModelContexts() {
    // 为每个推理 worker 分别创建模型上下文。
    // 这样每个线程都能独占自己的 RKNN context 和输入缓冲区。
    for (int i = 0; i < options_.infer_thread_count; ++i) {
        memset(&infer_contexts_[i], 0, sizeof(rknn_app_context_t));

        int ret = UseYolov8Model(options_)
                      ? init_yolov8_model(options_.model_path.c_str(), &infer_contexts_[i])
                      : init_yolov5_model(options_.model_path.c_str(), &infer_contexts_[i]);  // 当前 worker 的模型初始化结果。
        if (ret != 0) {
            printf("init_%s_model fail! worker=%d ret=%d\n",
                   UseYolov8Model(options_) ? "yolov8" : "yolov5",
                   i,
                   ret);
            return false;
        }

        // 按 worker 序号尽量把推理任务分散到不同 NPU 核心。
        // 即使设置失败，这里也不直接中断，让程序仍有机会继续运行。
        rknn_core_mask core_mask = SelectCoreMask(i, options_.infer_thread_count);  // 尽量为当前 worker 选择一个分散的 NPU 核心掩码。
        ret = rknn_set_core_mask(infer_contexts_[i].rknn_ctx, core_mask);
        (void)ret;
    }
    return true;
}

// 统一释放已加载的模型资源和相关上下文内存
void PipelineApp::ReleaseModelContexts() {
    // 无论程序是正常结束还是中途失败，都统一走这里释放模型资源。
    for (size_t i = 0; i < infer_contexts_.size(); ++i) {
        int ret = UseYolov8Model(options_)
                      ? release_yolov8_model(&infer_contexts_[i])
                      : release_yolov5_model(&infer_contexts_[i]);  // 当前上下文的资源释放结果。
        if (ret != 0) {
            printf("release_%s_model fail! ret=%d\n",
                   UseYolov8Model(options_) ? "yolov8" : "yolov5",
                   ret);
        }
    }
}

// 流水线主运行入口：负责探测码流、初始化各阶段模块、启动并发处理线程，并进行统一回收
int PipelineApp::Run() {
    // OpenCV 默认可能会再开内部线程。
    // 这里主动限制线程行为，避免与我们自己的流水线线程叠加竞争。
    cv::setNumThreads(1);
    cv::setUseOptimized(true);

    SignalWatcher signal_watcher(&runtime_, &queues_);  // 负责把系统信号转换为应用内部的停止请求。
    if (!signal_watcher.Start()) {
        printf("signal: graceful shutdown watcher init failed\n");
        return -1;
    }

    bool model_contexts_need_release = false;  // 标记后续清理阶段是否需要释放 RKNN 模型上下文。
    bool post_process_initialized = false;  // 标记后处理模块是否已经完成初始化。
    // 内部 Lambda：用于统一处理资源的清理释放并返回最终的退出状态码
    auto CleanupAndReturn = [&](int code) -> int {
        if (post_process_initialized) {
            deinit_post_process();
            post_process_initialized = false;
        }
        if (model_contexts_need_release) {
            ReleaseModelContexts();
            model_contexts_need_release = false;
        }
        if (signal_watcher.interrupted()) {
            return signal_watcher.ExitCode();
        }
        return code;
    };
    // 内部 Lambda：检查当前运行时是否已经发出了停止请求
    auto StopRequested = [&]() -> bool {
        return runtime_.stop_requested.load(std::memory_order_relaxed);
    };

    // 第一步：先探测输入流信息。
    // 这里使用一个临时 demuxer，只负责读取码流参数，
    // 比如编码类型、分辨率和 codec parameters，供后面的解码器初始化使用。
    MediaStreamInfo stream_info;  // 保存探测到的视频流信息，供解码器和后续阶段复用。
    {
        std::unique_ptr<FfmpegDemuxer> probe_demuxer(new FfmpegDemuxer());  // 只用于探测流参数的临时 demuxer。
        probe_demuxer->SetInterruptFlag(&runtime_.stop_requested);
        if (!probe_demuxer->Open(options_.input, &stream_info)) {
            if (!signal_watcher.interrupted()) {
                printf("demuxer open failed for input=%s\n", options_.input.c_str());
            }
            return CleanupAndReturn(-1);
        }
    }
    if (StopRequested()) {
        return CleanupAndReturn(0);
    }

    // 第二步：先用探测得到的流信息初始化解码阶段。
    // 这样可以把“输入流打开”和“FFmpeg rkmpp 解码器初始化”两个动作拆开处理，
    // 出现问题时更容易定位，也更符合当前工程的初始化顺序设计。
    DecodeStage decode_stage(stream_info, &queues_, &runtime_);  // 先用探测到的流参数准备解码阶段。
    if (!decode_stage.Init()) {
        printf("decode stage init failed\n");
        return CleanupAndReturn(-1);
    }
    if (StopRequested()) {
        return CleanupAndReturn(0);
    }

    // 第三步：重新创建真正用于采集线程的 demuxer。
    // 上面的临时 demuxer 已经完成探测职责，这里重新打开作为正式输入通道。
    std::unique_ptr<FfmpegDemuxer> demuxer(new FfmpegDemuxer());  // 正式交给采集阶段长期持有的 demuxer。
    demuxer->SetInterruptFlag(&runtime_.stop_requested);
    if (!demuxer->Open(options_.input, &stream_info)) {
        if (!signal_watcher.interrupted()) {
            printf("demuxer reopen failed for input=%s\n", options_.input.c_str());
        }
        return CleanupAndReturn(-1);
    }
    if (StopRequested()) {
        return CleanupAndReturn(0);
    }

    // 第四步：构造采集和渲染阶段对象。
    // 推理阶段对象会在模型上下文准备好之后再创建。
    CaptureStage capture_stage(options_, std::move(demuxer), &queues_, &runtime_);  // 负责持续从输入源读取压缩包。
    RenderStage render_stage(options_, &queues_, &runtime_);  // 负责消费推理结果并执行显示或统计输出。

    // 第五步：初始化所有推理上下文。
    // 这里放在解码器准备完成之后，有助于保持底层组件初始化顺序稳定。
    model_contexts_need_release = true;
    if (!InitModelContexts()) {
        return CleanupAndReturn(-1);
    }
    if (StopRequested()) {
        return CleanupAndReturn(0);
    }

    // 第六步：初始化后处理模块。
    // 这里放到模型初始化之后，便于按实际模型类别数校验标签数量。
    if (init_post_process(options_.labels_path.c_str(),
                          infer_contexts_.empty() ? 0 : infer_contexts_[0].class_count) != 0) {
        printf("init_post_process fail!\n");
        return CleanupAndReturn(-1);
    }
    post_process_initialized = true;
    if (StopRequested()) {
        return CleanupAndReturn(0);
    }

    // 第七步：为每个推理 worker 创建一个独立的阶段对象。
    std::vector<std::unique_ptr<InferenceStage> > infer_stages;  // 每个推理 worker 对应一个独立阶段对象。
    infer_stages.reserve(static_cast<size_t>(options_.infer_thread_count));
    for (int i = 0; i < options_.infer_thread_count; ++i) {
        infer_stages.emplace_back(
            new InferenceStage(i, options_, &infer_contexts_[i], &queues_, &runtime_));
    }

    // 第八步：启动各阶段线程。
    // 线程角色分别是：
    // - 1 个采集线程
    // - 1 个解码线程
    // - N 个推理线程
    // - 1 个渲染线程
    std::thread capture_thread(&CaptureStage::Run, &capture_stage);  // 采集线程。
    std::thread decode_thread(&DecodeStage::Run, &decode_stage);  // 解码线程。

    std::vector<std::thread> infer_threads;  // 推理线程集合。
    infer_threads.reserve(static_cast<size_t>(options_.infer_thread_count));
    for (int i = 0; i < options_.infer_thread_count; ++i) {
        infer_threads.push_back(std::thread(&InferenceStage::Run, infer_stages[i].get()));
    }

    std::thread render_thread(&RenderStage::Run, &render_stage);  // 渲染线程。

    // 第九步：等待所有线程退出。
    // 只有所有阶段都自然结束或收到停止信号后，程序才进入统一清理流程。
    if (capture_thread.joinable()) {
        capture_thread.join();
    }
    if (decode_thread.joinable()) {
        decode_thread.join();
    }
    for (size_t i = 0; i < infer_threads.size(); ++i) {
        if (infer_threads[i].joinable()) {
            infer_threads[i].join();
        }
    }
    if (render_thread.joinable()) {
        render_thread.join();
    }

    // 第十步：统一释放模型和后处理资源。
    return CleanupAndReturn(0);
}

}  // namespace rknn_demo
