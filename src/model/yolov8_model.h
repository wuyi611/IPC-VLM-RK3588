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

#ifndef RKNN_YOLOV5_DEMO_MODEL_YOLOV8_MODEL_H_
#define RKNN_YOLOV5_DEMO_MODEL_YOLOV8_MODEL_H_

// 文件说明：
// 定义 YOLOv8 常规 RKNN 模型调用接口。

#include "model/model_context.h"
#include "model/postprocess.h"

// 加载并初始化 YOLOv8 RKNN 模型。
// 这里会读取模型文件、查询 tensor 属性，并分配可复用的输入缓冲。
int init_yolov8_model(const char *model_path, rknn_app_context_t *app_ctx);

// 释放 YOLOv8 模型上下文相关资源。
int release_yolov8_model(rknn_app_context_t *app_ctx);

// 执行一次 YOLOv8 推理。
// 输入原始图像，输出检测框，并回填内部各阶段耗时统计。
int inference_yolov8_model(rknn_app_context_t *app_ctx,
                           image_buffer_t *img,
                           object_detect_result_list *od_results,
                           InferenceProfile *profile);

#endif  // RKNN_YOLOV5_DEMO_MODEL_YOLOV8_MODEL_H_
