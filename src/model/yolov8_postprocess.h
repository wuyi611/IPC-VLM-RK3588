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

#ifndef RKNN_YOLOV5_DEMO_MODEL_YOLOV8_POSTPROCESS_H_
#define RKNN_YOLOV5_DEMO_MODEL_YOLOV8_POSTPROCESS_H_

// 文件说明：
// 声明 YOLOv8 检测头对应的后处理接口。

#include "model/model_context.h"
#include "model/postprocess.h"

// 对 YOLOv8 模型输出做后处理，生成最终检测框列表。
// 当前针对 RK 常见的三尺度检测头输出格式。
int post_process_yolov8(rknn_app_context_t *app_ctx,
                        void *outputs,
                        letterbox_t *letter_box,
                        float conf_threshold,
                        float nms_threshold,
                        object_detect_result_list *od_results);

#endif  // RKNN_YOLOV5_DEMO_MODEL_YOLOV8_POSTPROCESS_H_
