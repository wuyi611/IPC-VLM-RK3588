#ifndef RKNN_YOLOV5_DEMO_MODEL_RKNN_INPUT_BUFFER_UTILS_H_
#define RKNN_YOLOV5_DEMO_MODEL_RKNN_INPUT_BUFFER_UTILS_H_

#include "model/model_context.h"

// 文件说明：
// 统一封装基于 DMA-BUF 的 RKNN 输入缓冲初始化与释放逻辑。

// 为给定模型上下文创建并绑定一块可复用的 RKNN 输入 DMA-BUF。
int InitRknnInputBuffer(rknn_app_context_t *app_ctx);

// 释放 `InitRknnInputBuffer` 创建的输入缓冲及相关 RKNN 内存对象。
void ReleaseRknnInputBuffer(rknn_app_context_t *app_ctx);

#endif  // RKNN_YOLOV5_DEMO_MODEL_RKNN_INPUT_BUFFER_UTILS_H_
