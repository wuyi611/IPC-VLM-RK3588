#include "app/app_options.h"
#include "app/pipeline_app.h"

// 文件说明：
// 这是程序入口文件，只负责解析启动参数并把控制权交给 `PipelineApp`。

int main(int argc, char **argv) {
    // `main` 只负责最核心的启动流程:
    // 1. 解析命令行参数.
    // 2. 参数非法时打印帮助并退出.
    // 3. 创建应用总控对象并启动整条流水线.
    rknn_demo::AppOptions options;

    // 将命令行参数统一整理到一个配置对象中,
    // 后续模块只读取 `options`, 不再直接操作 `argv`.
    if (!rknn_demo::AppOptionsParser::Parse(argc, argv, &options)) {
        // 启动阶段参数不正确时, 直接输出用法说明并停止,
        // 避免继续初始化解码器, 模型和渲染资源.
        rknn_demo::AppOptionsParser::PrintUsage(argv[0]);
        return -1;
    }

    // `PipelineApp` 负责完整的应用生命周期管理,
    // 包括资源初始化, 阶段创建, 线程启动和最终清理.
    rknn_demo::PipelineApp app(options);

    // 进入 `Run()` 后, 程序将按
    // "采集 -> 解码 -> 推理 -> 渲染" 的流水线运行.
    return app.Run();
}
