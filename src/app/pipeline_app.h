#ifndef RKNN_YOLOV5_DEMO_APP_PIPELINE_APP_H_
#define RKNN_YOLOV5_DEMO_APP_PIPELINE_APP_H_

#include <vector>

#include "app/app_options.h"
#include "model/model_context.h"
#include "pipeline/pipeline_queues.h"
#include "pipeline/pipeline_types.h"

// 文件说明：
// 定义整个应用的总控类。它负责管理共享资源，并把
// 采集、解码、推理、渲染四个阶段按统一流程组织起来。

namespace rknn_demo {

// `PipelineApp` 是应用层编排器。
// `main()` 只需要负责参数解析和对象创建，真正的初始化顺序、
// 线程启动顺序、资源回收顺序都集中在这个类中管理。
class PipelineApp {
public:
    // 使用启动参数构造总控对象。
    // 构造阶段只做轻量成员初始化，不执行真正的模型加载和线程启动。
    explicit PipelineApp(const AppOptions &options);

    // 执行完整的应用生命周期：
    // 1. 初始化后处理和模型资源；
    // 2. 探测输入流并初始化解码器；
    // 3. 创建并启动各阶段线程；
    // 4. 等待线程退出并统一释放资源。
    // 返回 0 表示正常结束，返回负值表示启动或运行失败。
    // 如果运行中收到退出信号，则返回值会转换为标准的信号退出码。
    int Run();

private:
    // 为每个推理 worker 初始化一个独立的 RKNN 上下文，
    // 并尽量把不同 worker 绑定到不同的 NPU 核心上。
    bool InitModelContexts();

    // 释放所有推理上下文。
    // 即使前面初始化过程中部分失败，也允许统一调用这里做兜底回收。
    void ReleaseModelContexts();

    // 启动时解析得到的只读配置。
    // 整个运行过程中所有阶段都以它为基础读取参数，
    // 避免在多个阶段之间重复传递零散参数。
    const AppOptions options_;

    // 流水线各阶段之间共享的有界队列集合。
    // 采集、解码、推理、渲染线程都通过它交换数据。
    PipelineQueues queues_;

    // 全局运行时状态。
    // 包含停止标志、活跃推理线程计数和统计信息。
    PipelineRuntime runtime_;

    // 推理上下文数组。
    // 每个推理线程独占一个 `rknn_app_context_t`，避免多线程争抢同一模型上下文。
    std::vector<rknn_app_context_t> infer_contexts_;
};

}  // namespace rknn_demo

#endif  // RKNN_YOLOV5_DEMO_APP_PIPELINE_APP_H_
