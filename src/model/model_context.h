// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef RKNN_YOLOV5_DEMO_MODEL_MODEL_CONTEXT_H_
#define RKNN_YOLOV5_DEMO_MODEL_MODEL_CONTEXT_H_

// 文件说明：
// 抽取 RKNN 检测模型共用的上下文和性能统计结构，供 YOLOv5 / YOLOv8 复用。

#include "common.h"
#include "rknn_api.h"

// 保存一次 RKNN 检测模型运行所需的全部上下文信息。
struct rknn_app_context_t {
    // RKNN 运行时上下文句柄。
    rknn_context rknn_ctx;

    // 模型输入输出 tensor 的数量信息。
    rknn_input_output_num io_num;

    // 输入 tensor 属性数组。
    rknn_tensor_attr *input_attrs;

    // 输出 tensor 属性数组。
    rknn_tensor_attr *output_attrs;

    // 模型输入通道数。
    int model_channel;

    // 模型输入宽度。
    int model_width;

    // 模型输入高度。
    int model_height;

    // 模型类别数。
    // 由模型输出 tensor 运行时推导得到，不再固定假设为 80 类。
    int class_count;

    // 当前模型输出是否为量化格式。
    bool is_quant;

    // 可复用的模型输入缓冲区。
    // 每个推理线程独占一块 DMA-BUF，供 RGA 输出和 RKNN 输入复用。
    image_buffer_t model_input_buffer;

    // 绑定到 RKNN 输入 tensor 的外部内存描述。
    rknn_tensor_mem *input_tensor_mem;

    // `rknn_set_io_mem` 使用的输入属性副本。
    rknn_tensor_attr input_io_attr;
};

// 记录单帧推理内部各阶段耗时。
struct InferenceProfile {
    // 预处理耗时。
    double preprocess_ms;

    // NPU 推理耗时。
    double npu_ms;

    // 后处理耗时。
    double postprocess_ms;

    // 总耗时。
    double total_ms;

    // 构造时把所有耗时清零。
    InferenceProfile()
        : preprocess_ms(0.0),
          npu_ms(0.0),
          postprocess_ms(0.0),
          total_ms(0.0) {}
};

#endif  // RKNN_YOLOV5_DEMO_MODEL_MODEL_CONTEXT_H_
