#include "app/app_options.h"

#include <stdio.h>
#include <stdlib.h>

// 文件说明：
// 实现应用启动参数的默认值设置、命令行解析和帮助信息打印。

namespace rknn_demo {

namespace {

// 这里只放当前 `.cc` 文件内部使用的辅助函数，
// 不需要对其他源文件暴露。

// 根据目标平台返回默认推理线程数。
// RK3588 为三核 NPU，因此在 aarch64 下默认给 3；
// 其他平台先保守地使用 1。
int DefaultInferThreadCount() {
#if defined(__aarch64__)
    return 3;
#else
    return 1;
#endif
}

// 将字符串解析为正整数。
// 解析成功时把结果写入 `value` 并返回 `true`；
// 输入非法、不是整数或超范围时返回 `false`。
bool ParsePositiveInt(const char *text, int *value) {
    char *end = NULL;
    long parsed = strtol(text, &end, 10);
    if (end == text || *end != '\0' || parsed <= 0 || parsed > 1024) {
        return false;
    }
    *value = static_cast<int>(parsed);
    return true;
}

// 将字符串解析为非负整数。
bool ParseNonNegativeInt(const char *text, int *value) {
    char *end = NULL;
    long parsed = strtol(text, &end, 10);
    if (end == text || *end != '\0' || parsed < 0 || parsed > 1024 * 1024) {
        return false;
    }
    *value = static_cast<int>(parsed);
    return true;
}

// 解析模型类型。
bool ParseModelType(const char *text, std::string *model_type) {
    if (text == NULL || model_type == NULL) {
        return false;
    }

    std::string value = text;
    if (value != "yolov5" && value != "yolov8") {
        return false;
    }

    *model_type = value;
    return true;
}

}     // namespace

bool AppOptionsParser::Parse(int argc, char **argv, AppOptions *options) {
    // 程序至少需要两个位置参数：
    // 1. 模型路径
    // 2. 视频文件路径或 RTSP 地址
    if (argc < 3 || options == NULL) {
        return false;
    }

    // 前两个位置参数固定分别表示模型路径和输入源。
    options->model_path = argv[1];
    options->input = argv[2];
    options->model_type = "yolov5";
    options->labels_path.clear();

    // 先写入一组默认配置，
    // 用户如果显式传入可选参数，再覆盖这些默认值。
    options->infer_thread_count = DefaultInferThreadCount();
    options->capture_queue_size = 3;
    options->decode_queue_size = 3;
    options->render_queue_size = 3;
    options->stats_interval = 30;
    options->enable_display = true;
    options->log_detections = false;
    options->enable_llm = false;
    options->llm_config_path = "src/cloud/providers/config.json";
    options->llm_queue_size = 1;
    options->llm_sample_every = 30;
    options->llm_min_interval_ms = 2000;
    options->llm_image_width = 640;
    options->llm_image_height = 360;
    options->llm_output_dir = "llm_samples";

    // 从第 3 个参数开始解析所有可选项。
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--infer-threads" && i + 1 < argc) {
            // 设置推理线程数量。
            if (!ParsePositiveInt(argv[++i], &options->infer_thread_count)) {
                return false;
            }
        } else if (arg == "--capture-queue" && i + 1 < argc) {
            // 设置采集阶段到解码阶段之间的队列深度。
            if (!ParsePositiveInt(argv[++i], &options->capture_queue_size)) {
                return false;
            }
        } else if (arg == "--decode-queue" && i + 1 < argc) {
            // 设置解码阶段到推理阶段之间的队列深度。
            if (!ParsePositiveInt(argv[++i], &options->decode_queue_size)) {
                return false;
            }
        } else if (arg == "--render-queue" && i + 1 < argc) {
            // 设置推理阶段到渲染阶段之间的队列深度。
            if (!ParsePositiveInt(argv[++i], &options->render_queue_size)) {
                return false;
            }
        } else if (arg == "--stats" && i + 1 < argc) {
            // 设置统计信息输出间隔。
            if (!ParsePositiveInt(argv[++i], &options->stats_interval)) {
                return false;
            }
        } else if (arg == "--model-type" && i + 1 < argc) {
            // 选择模型架构类型。
            if (!ParseModelType(argv[++i], &options->model_type)) {
                return false;
            }
        } else if (arg == "--labels" && i + 1 < argc) {
            // 标签文件必须显式传入，且不能为空字符串。
            options->labels_path = argv[++i];
            if (options->labels_path.empty()) {
                return false;
            }
        } else if (arg == "--no-display") {
            // 关闭本地显示，优先追求吞吐量。
            options->enable_display = false;
        } else if (arg == "--log-detections") {
            // 打开逐帧检测结果日志输出。
            options->log_detections = true;
        } else if (arg == "--enable-llm") {
            // 启用异步 LLM 场景理解支路。
            options->enable_llm = true;
        } else if (arg == "--llm-config" && i + 1 < argc) {
            // 指定 LLM 配置文件路径。
            options->llm_config_path = argv[++i];
            if (options->llm_config_path.empty()) {
                return false;
            }
        } else if (arg == "--llm-queue" && i + 1 < argc) {
            // 设置推理到 LLM 之间的队列深度。
            if (!ParsePositiveInt(argv[++i], &options->llm_queue_size)) {
                return false;
            }
        } else if (arg == "--llm-sample-every" && i + 1 < argc) {
            // 每多少帧送一次 LLM。
            if (!ParsePositiveInt(argv[++i], &options->llm_sample_every)) {
                return false;
            }
        } else if (arg == "--llm-min-interval-ms" && i + 1 < argc) {
            // 两次 LLM 请求之间的最小时间间隔。
            if (!ParseNonNegativeInt(argv[++i], &options->llm_min_interval_ms)) {
                return false;
            }
        } else if (arg == "--llm-image-width" && i + 1 < argc) {
            // LLM 抽样图的缩放宽度。
            if (!ParsePositiveInt(argv[++i], &options->llm_image_width)) {
                return false;
            }
        } else if (arg == "--llm-image-height" && i + 1 < argc) {
            // LLM 抽样图的缩放高度。
            if (!ParsePositiveInt(argv[++i], &options->llm_image_height)) {
                return false;
            }
        } else if (arg == "--llm-output-dir" && i + 1 < argc) {
            // LLM 抽样图的临时输出目录。
            options->llm_output_dir = argv[++i];
            if (options->llm_output_dir.empty()) {
                return false;
            }
        } else {
            // 参数未知，或者缺少所需取值时，直接视为解析失败。
            return false;
        }
    }

    // 标签文件现在必须由用户显式指定，不再回退到默认 COCO 标签。
    if (options->labels_path.empty()) {
        return false;
    }

    return true;
}

void AppOptionsParser::PrintUsage(const char *program) {
    // 第一行给出启动命令的总格式。
    printf("%s <model path> <video path|rtsp url> --labels <label path> [options]\n", program);

    // 给出本地文件输入和 RTSP 输入两种常见示例。
    printf("Usage: %s yolov5s.rknn ./video.mp4 --labels ./labels.txt --no-display\n", program);
    printf("Usage: %s yolov8s.rknn rtsp://user:pass@192.168.8.105:554/Streaming/Channels/101 --labels ./labels.txt --model-type yolov8\n", program);

    // 列出当前支持的全部可选参数。
    printf("Options:\n");
    printf("  --infer-threads N   Number of inference threads\n");
    printf("  --capture-queue N   Capture -> decode queue depth\n");
    printf("  --decode-queue N    Decode -> inference queue depth\n");
    printf("  --render-queue N    Inference -> render queue depth\n");
    printf("  --stats N           Print stats every N rendered frames\n");
    printf("  --model-type X      Model type: yolov5 | yolov8\n");
    printf("  --labels PATH       Label file path (required)\n");
    printf("  --no-display        Disable local rendering for max throughput\n");
    printf("  --log-detections    Print every detection result\n");
    printf("  --enable-llm        Enable async LLM scene understanding\n");
    printf("  --llm-config PATH   LLM config file path\n");
    printf("  --llm-queue N       Inference -> LLM queue depth\n");
    printf("  --llm-sample-every N  Sample every N frames for LLM\n");
    printf("  --llm-min-interval-ms N  Min interval between LLM requests\n");
    printf("  --llm-image-width N   LLM sample image width\n");
    printf("  --llm-image-height N  LLM sample image height\n");
    printf("  --llm-output-dir PATH Temp directory for LLM sample images\n");
}

}  // namespace rknn_demo
