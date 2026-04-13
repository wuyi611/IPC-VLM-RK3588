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
// 实现 YOLOv8 后处理，包括 DFL 解码、排序、NMS 和坐标还原。

#include "model/yolov8_postprocess.h"

#include <algorithm>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

namespace {

// 把坐标限制在有效图像范围内，避免映射回原图后越界。
inline int ClampCoord(float val, int min, int max) {
    return val > min ? (val < max ? val : max) : min;
}

// 计算两个候选框的 IoU。
static float CalculateOverlap(float xmin0,
                              float ymin0,
                              float xmax0,
                              float ymax0,
                              float xmin1,
                              float ymin1,
                              float xmax1,
                              float ymax1) {
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0f);  // 交集宽度。
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0f);  // 交集高度。
    float intersection = w * h;  // 交集面积。
    float union_area = (xmax0 - xmin0 + 1.0f) * (ymax0 - ymin0 + 1.0f) +
                       (xmax1 - xmin1 + 1.0f) * (ymax1 - ymin1 + 1.0f) -
                       intersection;  // 并集面积。
    return union_area <= 0.0f ? 0.0f : (intersection / union_area);
}

// 按类别执行 NMS，把被抑制的候选框标记成 -1。
static int Nms(int valid_count,
               std::vector<float> &boxes,
               const std::vector<int> &class_ids,
               std::vector<int> &order,
               int filter_id,
               float threshold) {
    for (int i = 0; i < valid_count; ++i) {
        int n = order[i];  // 当前保留下来的高分候选框索引。
        if (n == -1 || class_ids[n] != filter_id) {
            continue;
        }
        for (int j = i + 1; j < valid_count; ++j) {
            int m = order[j];  // 需要和 `n` 比较 IoU 的后续候选框索引。
            if (m == -1 || class_ids[m] != filter_id) {
                continue;
            }
            float xmin0 = boxes[n * 4 + 0];
            float ymin0 = boxes[n * 4 + 1];
            float xmax0 = boxes[n * 4 + 0] + boxes[n * 4 + 2];
            float ymax0 = boxes[n * 4 + 1] + boxes[n * 4 + 3];

            float xmin1 = boxes[m * 4 + 0];
            float ymin1 = boxes[m * 4 + 1];
            float xmax1 = boxes[m * 4 + 0] + boxes[m * 4 + 2];
            float ymax1 = boxes[m * 4 + 1] + boxes[m * 4 + 3];

            if (CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1) > threshold) {
                order[j] = -1;
            }
        }
    }
    return 0;
}

// 对候选框置信度做降序排序，同时保持索引同步变化。
static int QuickSortIndiceInverse(std::vector<float> &input,
                                  int left,
                                  int right,
                                  std::vector<int> &indices) {
    float key;  // 当前分区使用的基准分数。
    int key_index;  // 基准分数对应的原始候选框索引。
    int low = left;  // 左侧扫描指针。
    int high = right;  // 右侧扫描指针。
    if (left < right) {
        key_index = indices[left];
        key = input[left];
        while (low < high) {
            while (low < high && input[high] <= key) {
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key) {
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        QuickSortIndiceInverse(input, left, low - 1, indices);
        QuickSortIndiceInverse(input, low + 1, right, indices);
    }
    return low;
}

// 把值裁剪到给定浮点区间。
inline int32_t ClipToRange(float val, float min, float max) {
    float clipped = val <= min ? min : (val >= max ? max : val);  // 裁剪后的浮点值。
    return clipped;
}

// 把浮点阈值量化到 int8，便于和量化输出直接比较。
static int8_t QuantizeToI8(float f32, int32_t zp, float scale) {
    float dst_val = (f32 / scale) + zp;  // 按量化参数换算后的整数值。
    return static_cast<int8_t>(ClipToRange(dst_val, -128, 127));
}

// 把浮点阈值量化到 uint8，便于和量化输出直接比较。
static uint8_t QuantizeToU8(float f32, int32_t zp, float scale) {
    float dst_val = (f32 / scale) + zp;  // 按量化参数换算后的整数值。
    return static_cast<uint8_t>(ClipToRange(dst_val, 0, 255));
}

// 把量化后的 int8 数值反量化回浮点。
static float DequantizeI8(int8_t qnt, int32_t zp, float scale) {
    return (static_cast<float>(qnt) - static_cast<float>(zp)) * scale;
}

// 把量化后的 uint8 数值反量化回浮点。
static float DequantizeU8(uint8_t qnt, int32_t zp, float scale) {
    return (static_cast<float>(qnt) - static_cast<float>(zp)) * scale;
}

// DFL 解码：
// 把每个边界上离散分布的 logits 还原成连续坐标偏移。
static void ComputeDfl(const float *tensor, int dfl_len, float *box) {
    for (int b = 0; b < 4; ++b) {
        float exp_sum = 0.0f;  // 当前边界所有 bin 的指数和。
        float acc_sum = 0.0f;  // softmax 加权后的期望距离。
        std::vector<float> exp_t(static_cast<size_t>(dfl_len), 0.0f);  // 暂存每个 bin 的 exp(logit)。
        for (int i = 0; i < dfl_len; ++i) {
            exp_t[i] = expf(tensor[i + b * dfl_len]);
            exp_sum += exp_t[i];
        }
        for (int i = 0; i < dfl_len; ++i) {
            acc_sum += exp_t[i] / exp_sum * i;
        }
        box[b] = acc_sum;
    }
}

// 处理 uint8 量化输出，解码单个尺度上的候选框。
static int ProcessU8(uint8_t *box_tensor,
                     int32_t box_zp,
                     float box_scale,
                     uint8_t *score_tensor,
                     int32_t score_zp,
                     float score_scale,
                     uint8_t *score_sum_tensor,
                     int32_t score_sum_zp,
                     float score_sum_scale,
                     int grid_h,
                     int grid_w,
                     int stride,
                     int dfl_len,
                     int class_count,
                     std::vector<float> &boxes,
                     std::vector<float> &scores,
                     std::vector<int> &class_ids,
                     float threshold) {
    // 当前尺度上通过阈值筛选后保留下来的候选框数量。
    int valid_count = 0;  // 当前尺度上通过阈值筛选的候选框数量。
    int grid_len = grid_h * grid_w;  // 单个通道的网格元素总数。
    uint8_t score_thres_u8 = QuantizeToU8(threshold, score_zp, score_scale);  // 量化后的类别分数阈值。
    uint8_t score_sum_thres_u8 = QuantizeToU8(threshold, score_sum_zp, score_sum_scale);  // 量化后的 `score_sum` 预筛阈值。

    for (int i = 0; i < grid_h; ++i) {
        for (int j = 0; j < grid_w; ++j) {
            // 当前网格位置在线性张量中的起始偏移。
            int offset = i * grid_w + j;  // 当前网格位置在线性张量中的起始偏移。
            int max_class_id = -1;  // 当前网格位置分数最高的类别 ID。
            // `score_sum` 是 RK 优化导出里额外增加的一路快速过滤分支。
            if (score_sum_tensor != NULL && score_sum_tensor[offset] < score_sum_thres_u8) {
                continue;
            }

            // 在当前网格位置上扫描所有类别，选出最大分类分数。
            uint8_t max_score = static_cast<uint8_t>(-score_zp);  // 当前网格位置最高类别分数。
            for (int c = 0; c < class_count; ++c) {
                if ((score_tensor[offset] > score_thres_u8) && (score_tensor[offset] > max_score)) {
                    max_score = score_tensor[offset];
                    max_class_id = c;
                }
                offset += grid_len;
            }

            // 只有分类分数超过阈值的点才继续做 DFL 解码和框恢复。
            if (max_score > score_thres_u8) {
                offset = i * grid_w + j;
                std::vector<float> before_dfl(static_cast<size_t>(dfl_len * 4), 0.0f);  // 反量化后的四条边 DFL logits。
                float box[4];  // DFL 解码后的 `ltrb` 距离。
                for (int k = 0; k < dfl_len * 4; ++k) {
                    before_dfl[k] = DequantizeU8(box_tensor[offset], box_zp, box_scale);
                    offset += grid_len;
                }
                ComputeDfl(before_dfl.data(), dfl_len, box);

                // YOLOv8 使用 ltrb 形式的距离回归，再换算成当前特征层上的 xyxy。
                float x1 = (-box[0] + j + 0.5f) * stride;
                float y1 = (-box[1] + i + 0.5f) * stride;
                float x2 = (box[2] + j + 0.5f) * stride;
                float y2 = (box[3] + i + 0.5f) * stride;
                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(x2 - x1);
                boxes.push_back(y2 - y1);
                scores.push_back(DequantizeU8(max_score, score_zp, score_scale));
                class_ids.push_back(max_class_id);
                valid_count++;
            }
        }
    }
    return valid_count;
}

// 处理 int8 量化输出，解码单个尺度上的候选框。
static int ProcessI8(int8_t *box_tensor,
                     int32_t box_zp,
                     float box_scale,
                     int8_t *score_tensor,
                     int32_t score_zp,
                     float score_scale,
                     int8_t *score_sum_tensor,
                     int32_t score_sum_zp,
                     float score_sum_scale,
                     int grid_h,
                     int grid_w,
                     int stride,
                     int dfl_len,
                     int class_count,
                     std::vector<float> &boxes,
                     std::vector<float> &scores,
                     std::vector<int> &class_ids,
                     float threshold) {
    // 当前尺度上通过阈值筛选后保留下来的候选框数量。
    int valid_count = 0;  // 当前尺度上通过阈值筛选的候选框数量。
    int grid_len = grid_h * grid_w;  // 单个通道的网格元素总数。
    int8_t score_thres_i8 = QuantizeToI8(threshold, score_zp, score_scale);  // 量化后的类别分数阈值。
    int8_t score_sum_thres_i8 = QuantizeToI8(threshold, score_sum_zp, score_sum_scale);  // 量化后的 `score_sum` 预筛阈值。

    for (int i = 0; i < grid_h; ++i) {
        for (int j = 0; j < grid_w; ++j) {
            // 当前网格位置在线性张量中的起始偏移。
            int offset = i * grid_w + j;  // 当前网格位置在线性张量中的起始偏移。
            int max_class_id = -1;  // 当前网格位置分数最高的类别 ID。
            if (score_sum_tensor != NULL && score_sum_tensor[offset] < score_sum_thres_i8) {
                continue;
            }

            // 在当前网格位置上扫描所有类别，选出最大分类分数。
            int8_t max_score = static_cast<int8_t>(-score_zp);  // 当前网格位置最高类别分数。
            for (int c = 0; c < class_count; ++c) {
                if ((score_tensor[offset] > score_thres_i8) && (score_tensor[offset] > max_score)) {
                    max_score = score_tensor[offset];
                    max_class_id = c;
                }
                offset += grid_len;
            }

            // 只有分类分数超过阈值的点才继续做 DFL 解码和框恢复。
            if (max_score > score_thres_i8) {
                offset = i * grid_w + j;
                std::vector<float> before_dfl(static_cast<size_t>(dfl_len * 4), 0.0f);  // 反量化后的四条边 DFL logits。
                float box[4];  // DFL 解码后的 `ltrb` 距离。
                for (int k = 0; k < dfl_len * 4; ++k) {
                    before_dfl[k] = DequantizeI8(box_tensor[offset], box_zp, box_scale);
                    offset += grid_len;
                }
                ComputeDfl(before_dfl.data(), dfl_len, box);

                // YOLOv8 使用 ltrb 形式的距离回归，再换算成当前特征层上的 xyxy。
                float x1 = (-box[0] + j + 0.5f) * stride;
                float y1 = (-box[1] + i + 0.5f) * stride;
                float x2 = (box[2] + j + 0.5f) * stride;
                float y2 = (box[3] + i + 0.5f) * stride;
                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(x2 - x1);
                boxes.push_back(y2 - y1);
                scores.push_back(DequantizeI8(max_score, score_zp, score_scale));
                class_ids.push_back(max_class_id);
                valid_count++;
            }
        }
    }
    return valid_count;
}

// 处理浮点输出，逻辑与量化版一致，只是省掉反量化过程。
static int ProcessFp32(float *box_tensor,
                       float *score_tensor,
                       float *score_sum_tensor,
                       int grid_h,
                       int grid_w,
                       int stride,
                       int dfl_len,
                       int class_count,
                       std::vector<float> &boxes,
                       std::vector<float> &scores,
                       std::vector<int> &class_ids,
                       float threshold) {
    // 当前尺度上通过阈值筛选后保留下来的候选框数量。
    int valid_count = 0;  // 当前尺度上通过阈值筛选的候选框数量。
    int grid_len = grid_h * grid_w;  // 单个通道的网格元素总数。
    for (int i = 0; i < grid_h; ++i) {
        for (int j = 0; j < grid_w; ++j) {
            // 当前网格位置在线性张量中的起始偏移。
            int offset = i * grid_w + j;  // 当前网格位置在线性张量中的起始偏移。
            int max_class_id = -1;  // 当前网格位置分数最高的类别 ID。
            if (score_sum_tensor != NULL && score_sum_tensor[offset] < threshold) {
                continue;
            }

            // 在当前网格位置上扫描所有类别，选出最大分类分数。
            float max_score = 0.0f;  // 当前网格位置最高类别分数。
            for (int c = 0; c < class_count; ++c) {
                if ((score_tensor[offset] > threshold) && (score_tensor[offset] > max_score)) {
                    max_score = score_tensor[offset];
                    max_class_id = c;
                }
                offset += grid_len;
            }

            // 只有分类分数超过阈值的点才继续做 DFL 解码和框恢复。
            if (max_score > threshold) {
                offset = i * grid_w + j;
                std::vector<float> before_dfl(static_cast<size_t>(dfl_len * 4), 0.0f);  // 四条边的 DFL logits。
                float box[4];  // DFL 解码后的 `ltrb` 距离。
                for (int k = 0; k < dfl_len * 4; ++k) {
                    before_dfl[k] = box_tensor[offset];
                    offset += grid_len;
                }
                ComputeDfl(before_dfl.data(), dfl_len, box);

                float x1 = (-box[0] + j + 0.5f) * stride;
                float y1 = (-box[1] + i + 0.5f) * stride;
                float x2 = (box[2] + j + 0.5f) * stride;
                float y2 = (box[3] + i + 0.5f) * stride;
                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(x2 - x1);
                boxes.push_back(y2 - y1);
                scores.push_back(max_score);
                class_ids.push_back(max_class_id);
                valid_count++;
            }
        }
    }
    return valid_count;
}

}  // namespace

int post_process_yolov8(rknn_app_context_t *app_ctx,
                        void *outputs,
                        letterbox_t *letter_box,
                        float conf_threshold,
                        float nms_threshold,
                        object_detect_result_list *od_results) {
    rknn_output *output_tensors = static_cast<rknn_output *>(outputs);  // 统一转换后的 RKNN 输出数组。
    thread_local std::vector<float> filter_boxes;  // 候选框坐标缓存，按 `[x, y, w, h]` 顺序存储。
    thread_local std::vector<float> scores;  // 候选框分数缓存。
    thread_local std::vector<int> class_ids;  // 候选框类别 ID 缓存。
    thread_local std::vector<int> index_array;  // 排序后的候选框索引数组。

    // 三个尺度上累计保留下来的候选框数量。
    int valid_count = 0;  // 三个尺度累计保留下来的候选框数量。
    int model_in_w = app_ctx->model_width;  // 模型输入宽度。
    int model_in_h = app_ctx->model_height;  // 模型输入高度。
    // RK 优化导出的 YOLOv8 一般是三组输出，每组由 box / cls / score_sum 组成。
    int output_per_branch = app_ctx->io_num.n_output / 3;  // 每个尺度检测头占用的输出 tensor 数量。

    memset(od_results, 0, sizeof(*od_results));
    filter_boxes.clear();
    scores.clear();
    class_ids.clear();
    index_array.clear();

    // 逐个尺度处理检测头输出。
    for (int i = 0; i < 3; ++i) {
        int box_idx = i * output_per_branch;  // 当前尺度 box 分支的输出索引。
        int score_idx = box_idx + 1;  // 当前尺度分类分支的输出索引。
        const rknn_tensor_attr &box_attr = app_ctx->output_attrs[box_idx];  // box 分支 tensor 属性。
        const rknn_tensor_attr &score_attr = app_ctx->output_attrs[score_idx];  // 分类分支 tensor 属性。
        int dfl_channels = 0;  // box 分支在通道方向上的总长度。
        int class_count = 0;  // 当前尺度解析出的类别数。
        int grid_h = 0;  // 当前尺度网格高度。
        int grid_w = 0;  // 当前尺度网格宽度。

        // 同时兼容 NCHW / NHWC 的张量属性描述，推断当前尺度的网格尺寸和回归通道数。
        if (box_attr.fmt == RKNN_TENSOR_NHWC && box_attr.n_dims >= 4) {
            grid_h = box_attr.dims[1];
            grid_w = box_attr.dims[2];
            dfl_channels = box_attr.dims[3];
        } else {
            grid_h = box_attr.dims[2];
            grid_w = box_attr.dims[3];
            dfl_channels = box_attr.dims[1];
        }

        // 分类分支的通道数就是当前模型的类别数。
        if (score_attr.fmt == RKNN_TENSOR_NHWC && score_attr.n_dims >= 4) {
            class_count = score_attr.dims[3];
        } else {
            class_count = score_attr.dims[1];
        }
        if (app_ctx->class_count > 0) {
            class_count = app_ctx->class_count;
        }

        // 每条边的 DFL bin 数，通常是 16。
        int dfl_len = dfl_channels / 4;  // 每条边的 DFL bin 数。
        // 特征图步长由输入尺寸与当前网格高度的比值得到。
        int stride = model_in_h / grid_h;  // 当前尺度相对输入图的步长。
        void *score_sum = NULL;  // RK 优化导出时附带的快速预筛分支。
        int32_t score_sum_zp = 0;  // `score_sum` 的零点。
        float score_sum_scale = 1.0f;  // `score_sum` 的缩放因子。
        if (output_per_branch == 3) {
            score_sum = output_tensors[box_idx + 2].buf;
            score_sum_zp = app_ctx->output_attrs[box_idx + 2].zp;
            score_sum_scale = app_ctx->output_attrs[box_idx + 2].scale;
        }

        // 根据输出量化类型分流到对应解码路径。
        if (app_ctx->is_quant) {
            if (app_ctx->output_attrs[box_idx].type == RKNN_TENSOR_UINT8) {
                valid_count += ProcessU8(static_cast<uint8_t *>(output_tensors[box_idx].buf),
                                         app_ctx->output_attrs[box_idx].zp,
                                         app_ctx->output_attrs[box_idx].scale,
                                         static_cast<uint8_t *>(output_tensors[score_idx].buf),
                                         app_ctx->output_attrs[score_idx].zp,
                                         app_ctx->output_attrs[score_idx].scale,
                                         static_cast<uint8_t *>(score_sum),
                                         score_sum_zp,
                                         score_sum_scale,
                                         grid_h,
                                         grid_w,
                                         stride,
                                         dfl_len,
                                         class_count,
                                         filter_boxes,
                                         scores,
                                         class_ids,
                                         conf_threshold);
            } else {
                valid_count += ProcessI8(static_cast<int8_t *>(output_tensors[box_idx].buf),
                                         app_ctx->output_attrs[box_idx].zp,
                                         app_ctx->output_attrs[box_idx].scale,
                                         static_cast<int8_t *>(output_tensors[score_idx].buf),
                                         app_ctx->output_attrs[score_idx].zp,
                                         app_ctx->output_attrs[score_idx].scale,
                                         static_cast<int8_t *>(score_sum),
                                         score_sum_zp,
                                         score_sum_scale,
                                         grid_h,
                                         grid_w,
                                         stride,
                                         dfl_len,
                                         class_count,
                                         filter_boxes,
                                         scores,
                                         class_ids,
                                         conf_threshold);
            }
        } else {
            valid_count += ProcessFp32(static_cast<float *>(output_tensors[box_idx].buf),
                                       static_cast<float *>(output_tensors[score_idx].buf),
                                       static_cast<float *>(score_sum),
                                       grid_h,
                                       grid_w,
                                       stride,
                                       dfl_len,
                                       class_count,
                                       filter_boxes,
                                       scores,
                                       class_ids,
                                       conf_threshold);
        }
    }

    // 没有任何候选框通过阈值时，直接返回空结果。
    if (valid_count <= 0) {
        return 0;
    }

    // 建立排序索引，按分数从高到低做 NMS。
    for (int i = 0; i < valid_count; ++i) {
        index_array.push_back(i);
    }
    QuickSortIndiceInverse(scores, 0, valid_count - 1, index_array);

    // 记录当前帧中实际出现过的类别，只对这些类别执行 NMS。
    std::vector<bool> class_seen(static_cast<size_t>(app_ctx->class_count > 0
                                                         ? app_ctx->class_count
                                                         : (class_ids.empty()
                                                                ? 0
                                                                : (*std::max_element(class_ids.begin(), class_ids.end()) + 1))),
                                 false);
    for (size_t i = 0; i < class_ids.size(); ++i) {
        int cls = class_ids[i];
        if (cls >= 0 && static_cast<size_t>(cls) < class_seen.size()) {
            class_seen[cls] = true;
        }
    }

    for (size_t c = 0; c < class_seen.size(); ++c) {
        if (class_seen[c]) {
            Nms(valid_count, filter_boxes, class_ids, index_array, static_cast<int>(c), nms_threshold);
        }
    }

    // 把 letterbox 坐标映射回原图坐标，并填充最终检测结果。
    int last_count = 0;  // 最终写入输出结果数组的目标数。
    for (int i = 0; i < valid_count && last_count < OBJ_NUMB_MAX_SIZE; ++i) {
        if (index_array[i] == -1) {
            continue;
        }
        int n = index_array[i];  // 当前保留下来的候选框原始索引。
        float x1 = filter_boxes[n * 4 + 0] - letter_box->x_pad;  // 映射回原图前的左上角 x。
        float y1 = filter_boxes[n * 4 + 1] - letter_box->y_pad;  // 映射回原图前的左上角 y。
        float x2 = x1 + filter_boxes[n * 4 + 2];  // 映射回原图前的右下角 x。
        float y2 = y1 + filter_boxes[n * 4 + 3];  // 映射回原图前的右下角 y。

        od_results->results[last_count].box.left =
            static_cast<int>(ClampCoord(x1, 0, model_in_w) / letter_box->scale);
        od_results->results[last_count].box.top =
            static_cast<int>(ClampCoord(y1, 0, model_in_h) / letter_box->scale);
        od_results->results[last_count].box.right =
            static_cast<int>(ClampCoord(x2, 0, model_in_w) / letter_box->scale);
        od_results->results[last_count].box.bottom =
            static_cast<int>(ClampCoord(y2, 0, model_in_h) / letter_box->scale);
        od_results->results[last_count].prop = scores[i];
        od_results->results[last_count].cls_id = class_ids[n];
        last_count++;
    }

    od_results->count = last_count;
    return 0;
}
