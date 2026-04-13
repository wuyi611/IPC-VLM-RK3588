#ifndef RKNN_YOLOV5_DEMO_APP_APP_OPTIONS_H_
#define RKNN_YOLOV5_DEMO_APP_APP_OPTIONS_H_

#include <string>

// 文件说明：
// 定义应用启动参数结构以及命令行解析器接口。

namespace rknn_demo {

/*
 * AppOptions 负责承载整个程序的运行配置。
 *
 * 之所以单独抽出来，而不是继续塞在 main.cc 里，
 * 是因为它会被：
 * - 命令行解析器
 * - 流水线编排器
 * - 各个阶段线程
 *
 * 同时使用。集中定义以后，后续加新参数时不会四处改。
 */
struct AppOptions {
    // RKNN 模型文件路径, 例如 `.rknn` 模型文件.
    std::string model_path;

    // 输入源, 可以是本地视频文件路径, 也可以是 RTSP 地址.
    std::string input;

    // 检测模型类型, 当前支持 `yolov5` 或 `yolov8`.
    std::string model_type;

    // 标签文件路径，要求通过命令行显式传入。
    std::string labels_path;

    // 推理线程数量, 通常对应要启动多少个 NPU 推理 worker.
    int infer_thread_count;

    // 采集阶段到解码阶段之间的队列深度.
    int capture_queue_size;

    // 解码阶段到推理阶段之间的队列深度.
    int decode_queue_size;

    // 推理阶段到渲染阶段之间的队列深度.
    int render_queue_size;

    // 每处理多少帧打印一次统计信息.
    int stats_interval;

    // 是否启用本地显示输出.
    bool enable_display;

    // 是否打印每一帧的检测结果明细.
    bool log_detections;
};

/*
 * AppOptionsParser 专门负责命令行解析和帮助信息。
 *
 * 这样 main.cc 可以只保留“启动程序”的职责，
 * 不再掺杂参数解析细节。
 */
class AppOptionsParser {
public:
    // 解析命令行参数并填充 `options`.
    // 解析成功返回 `true`, 参数非法或缺失时返回 `false`.
    static bool Parse(int argc, char **argv, AppOptions *options);

    // 打印程序用法说明和支持的可选参数列表.
    static void PrintUsage(const char *program);
};

}  // namespace rknn_demo

#endif  // RKNN_YOLOV5_DEMO_APP_APP_OPTIONS_H_
