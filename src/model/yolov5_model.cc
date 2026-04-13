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

// 文件说明：
// 实现 YOLOv5 RKNN 模型的加载、输入准备、推理执行和资源释放。

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "common.h"
#include "file_utils.h"
#include "image_utils.h"
#include "model/rknn_input_buffer_utils.h"
#include "model/yolov5_model.h"

namespace {

// 把 `timeval` 转成微秒数，供推理耗时统计复用。
double GetProfileUs(const struct timeval &time_value) {
    return (time_value.tv_sec * 1000000.0 + time_value.tv_usec);
}

// 从 YOLOv5 输出 tensor 属性中推导类别数。
int InferYolov5ClassCount(const rknn_tensor_attr &output_attr) {
    int channels = 0;  // 当前输出分支在类别维度上的总通道数。
    if (output_attr.fmt == RKNN_TENSOR_NHWC && output_attr.n_dims >= 4) {
        channels = output_attr.dims[3];
    } else {
        channels = output_attr.dims[1];
    }

    if (channels <= 0 || channels % 3 != 0) {
        return -1;
    }

    int per_anchor_channels = channels / 3;  // 每个 anchor 对应的输出通道数。
    return per_anchor_channels > 5 ? (per_anchor_channels - 5) : -1;
}

}  // namespace

// 加载 YOLOv5 RKNN 模型，并初始化输入输出属性和可复用输入缓冲。
int init_yolov5_model(const char *model_path, rknn_app_context_t *app_ctx) {
    int ret;  // 通用返回值。
    int model_len = 0;  // 模型文件总字节数。
    char *model = NULL;  // 从磁盘读出的模型二进制数据。
    rknn_context ctx = 0;  // 本次初始化得到的 RKNN context。

    /*
     * 先把可复用的模型输入缓冲描述符清零。
     * 真正的缓冲会在拿到模型输入尺寸后一次性分配。
     */
    memset(&app_ctx->model_input_buffer, 0, sizeof(app_ctx->model_input_buffer));
    app_ctx->input_tensor_mem = NULL;
    memset(&app_ctx->input_io_attr, 0, sizeof(app_ctx->input_io_attr));

    // 读取 `.rknn` 模型文件到内存。
    model_len = read_data_from_file(model_path, &model);
    if (model_len <= 0 || model == NULL) {
        printf("load_model fail! path=%s\n", model_path);
        return -1;
    }

    // 创建 RKNN 运行时上下文。
    ret = rknn_init(&ctx, model, model_len, 0, NULL);
    free(model);
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    rknn_input_output_num io_num;  // 模型输入输出 tensor 的数量信息。
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }

    // 临时保存查询到的输入 tensor 属性。
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
    }

    // 临时保存查询到的输出 tensor 属性。
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
    }

    // 把本次初始化得到的核心上下文挂到应用上下文里。
    app_ctx->rknn_ctx = ctx;

    // 判断当前模型输出是否走量化后处理路径。
    if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC &&
        output_attrs[0].type != RKNN_TENSOR_FLOAT16) {
        app_ctx->is_quant = true;
    } else {
        app_ctx->is_quant = false;
    }

    // YOLOv5 每个尺度输出都按 3 个 anchor 展开，因此类别数可从通道数反推。
    app_ctx->class_count = InferYolov5ClassCount(output_attrs[0]);
    if (app_ctx->class_count <= 0) {
        printf("infer yolov5 class_count failed from output channels=%d\n",
               output_attrs[0].fmt == RKNN_TENSOR_NHWC ? output_attrs[0].dims[3]
                                                       : output_attrs[0].dims[1]);
        return -1;
    }

    app_ctx->io_num = io_num;
    app_ctx->input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx->output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    // 根据输入 tensor 排布方式推导模型输入宽高和通道数。
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        app_ctx->model_channel = input_attrs[0].dims[1];
        app_ctx->model_height = input_attrs[0].dims[2];
        app_ctx->model_width = input_attrs[0].dims[3];
    } else {
        app_ctx->model_height = input_attrs[0].dims[1];
        app_ctx->model_width = input_attrs[0].dims[2];
        app_ctx->model_channel = input_attrs[0].dims[3];
    }

    if (InitRknnInputBuffer(app_ctx) != 0) {
        printf("InitRknnInputBuffer fail!\n");
        return -1;
    }

    return 0;
}

// 释放 YOLOv5 模型占用的 tensor 属性、输入缓冲和 RKNN context。
int release_yolov5_model(rknn_app_context_t *app_ctx) {
    if (app_ctx->input_attrs != NULL) {
        free(app_ctx->input_attrs);
        app_ctx->input_attrs = NULL;
    }
    if (app_ctx->output_attrs != NULL) {
        free(app_ctx->output_attrs);
        app_ctx->output_attrs = NULL;
    }

    // context 销毁时把长期复用的输入缓冲一起释放。
    ReleaseRknnInputBuffer(app_ctx);
    if (app_ctx->rknn_ctx != 0) {
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    return 0;
}

// 执行一次 YOLOv5 推理，并回填检测结果和阶段耗时。
int inference_yolov5_model(rknn_app_context_t *app_ctx,
                           image_buffer_t *img,
                           object_detect_result_list *od_results,
                           InferenceProfile *profile) {
    int ret;  // 通用返回值。

    // 直接使用 context 里的长期输入缓冲，避免逐帧 malloc/free。
    image_buffer_t *dst_img = &app_ctx->model_input_buffer;
    letterbox_t letter_box;  // 保存预处理阶段生成的 letterbox 信息。

    rknn_output outputs[app_ctx->io_num.n_output];  // RKNN 输出描述数组。
    const float nms_threshold = NMS_THRESH;  // 默认 NMS 阈值。
    const float box_conf_threshold = BOX_THRESH;  // 默认目标置信度阈值。
    int bg_color = 114;  // letterbox 填充背景色。

    // 各阶段开始/结束时间戳，用于统计预处理、NPU 和后处理耗时。
    struct timeval preprocess_start;
    struct timeval preprocess_end;
    struct timeval npu_start;
    struct timeval npu_end;
    struct timeval post_start;
    struct timeval post_end;

    if ((!app_ctx) || !(img) || (!od_results)) {
        return -1;
    }

    memset(od_results, 0x00, sizeof(*od_results));
    if (profile != NULL) {
        *profile = InferenceProfile();
    }
    memset(&letter_box, 0, sizeof(letterbox_t));
    memset(outputs, 0, sizeof(outputs));

    if (dst_img->fd < 0 || app_ctx->input_tensor_mem == NULL) {
        printf("model input buffer is null\n");
        return -1;
    }

    /*
     * 预处理：
     * - 根据模型输入尺寸做 letterbox
     * - 优先走 RGA
     * - 同时完成颜色空间适配
     */
    gettimeofday(&preprocess_start, NULL);
    ret = convert_image_with_letterbox(img, dst_img, &letter_box, bg_color);
    gettimeofday(&preprocess_end, NULL);
    if (ret < 0) {
        printf("convert_image_with_letterbox fail! ret=%d\n", ret);
        return -1;
    }

    // 正式触发 NPU 推理。
    gettimeofday(&npu_start, NULL);
    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    gettimeofday(&npu_end, NULL);
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }

    // 取出模型输出。
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        outputs[i].index = i;
        outputs[i].want_float = (!app_ctx->is_quant);
    }
    ret = rknn_outputs_get(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs, NULL);
    if (ret < 0) {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        goto out;
    }

    // 后处理阶段会把输出 tensor 解码成最终的检测框列表。
    gettimeofday(&post_start, NULL);
    post_process(app_ctx, outputs, &letter_box, box_conf_threshold, nms_threshold, od_results);
    gettimeofday(&post_end, NULL);

    // RKNN 的输出需要显式释放。
    rknn_outputs_release(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs);

    if (profile != NULL) {
        profile->preprocess_ms = (GetProfileUs(preprocess_end) - GetProfileUs(preprocess_start)) / 1000.0;
        profile->npu_ms = (GetProfileUs(npu_end) - GetProfileUs(npu_start)) / 1000.0;
        profile->postprocess_ms = (GetProfileUs(post_end) - GetProfileUs(post_start)) / 1000.0;
        profile->total_ms = profile->preprocess_ms + profile->npu_ms + profile->postprocess_ms;
    }

out:
    return ret;
}
