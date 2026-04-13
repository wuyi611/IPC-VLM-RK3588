// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
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
// 实现 YOLOv5 后处理，包括候选框解码、排序、NMS 和标签加载。

#include "model/yolov5_model.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <vector>

// 标签名称表，初始化后按类别 ID 查询。
static std::vector<char *> labels;

// YOLOv5 三个检测头对应的 anchor 配置。
const int anchor[3][6] = {{10, 13, 16, 30, 33, 23},
                          {30, 61, 62, 45, 59, 119},
                          {116, 90, 156, 198, 373, 326}};

// 把浮点坐标裁剪到给定区间。
inline static int clamp(float val, int min, int max) {
    return val > min ? (val < max ? val : max) : min;
}

// 从标签文件中读取一行文本，并返回新分配的字符串缓冲区。
static char *readLine(FILE *fp, char *buffer, int *len) {
    int ch;  // 当前读取到的字符。
    int i = 0;  // 已写入缓冲区的字符个数。
    size_t buff_len = 0;  // 当前缓冲区长度。

    buffer = (char *)malloc(buff_len + 1);
    if (!buffer) {
        return NULL;
    }

    while ((ch = fgetc(fp)) != '\n' && ch != EOF) {
        buff_len++;
        void *tmp = realloc(buffer, buff_len + 1);
        if (tmp == NULL) {
            free(buffer);
            return NULL;
        }
        buffer = (char *)tmp;
        buffer[i] = (char)ch;
        i++;
    }
    buffer[i] = '\0';

    *len = static_cast<int>(buff_len);

    if (ch == EOF && (i == 0 || ferror(fp))) {
        free(buffer);
        return NULL;
    }
    return buffer;
}

// 读取整个标签文件，按行填充到动态标签表中。
static int readLines(const char *fileName, std::vector<char *> *lines) {
    FILE *file = fopen(fileName, "r");  // 打开的标签文件句柄。
    char *s = NULL;  // 单行标签文本缓冲。
    int n = 0;  // 单行标签长度。

    if (file == NULL || lines == NULL) {
        printf("Open %s fail!\n", fileName);
        return -1;
    }

    lines->clear();
    while ((s = readLine(file, s, &n)) != NULL) {
        lines->push_back(s);
    }
    fclose(file);
    return static_cast<int>(lines->size());
}

// 加载标签名称表，供类别 ID 到显示名称的映射使用。
static int loadLabelName(const char *locationFilename, std::vector<char *> *loaded_labels) {
    return readLines(locationFilename, loaded_labels);
}

// 计算两个检测框的 IoU，用于 NMS 判断是否需要抑制。
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
    float i = w * h;  // 交集面积。
    float u = (xmax0 - xmin0 + 1.0f) * (ymax0 - ymin0 + 1.0f) +
              (xmax1 - xmin1 + 1.0f) * (ymax1 - ymin1 + 1.0f) - i;  // 并集面积。
    return u <= 0.f ? 0.f : (i / u);
}

// 对指定类别执行 NMS，把被抑制的候选框索引标记为 -1。
static int nms(int validCount,
               std::vector<float> &outputLocations,
               std::vector<int> classIds,
               std::vector<int> &order,
               int filterId,
               float threshold) {
    for (int i = 0; i < validCount; ++i) {
        int n = order[i];  // 当前保留下来的高分候选框索引。
        if (n == -1 || classIds[n] != filterId) {
            continue;
        }
        for (int j = i + 1; j < validCount; ++j) {
            int m = order[j];  // 需要和 `n` 比较 IoU 的后续候选框索引。
            if (m == -1 || classIds[m] != filterId) {
                continue;
            }

            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
            float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);
            if (iou > threshold) {
                order[j] = -1;
            }
        }
    }
    return 0;
}

// 按置信度从高到低排序，同时保持索引数组和分数数组同步。
static int quick_sort_indice_inverse(std::vector<float> &input,
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
        quick_sort_indice_inverse(input, left, low - 1, indices);
        quick_sort_indice_inverse(input, low + 1, right, indices);
    }
    return low;
}

// sigmoid 激活函数。
static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// sigmoid 的反函数。
static float unsigmoid(float y) { return -1.0f * logf((1.0f / y) - 1.0f); }

// 将浮点值裁剪到指定区间，供量化转换时使用。
inline static int32_t __clip(float val, float min, float max) {
    float f = val <= min ? min : (val >= max ? max : val);  // 裁剪后的浮点值。
    return static_cast<int32_t>(f);
}

// 将浮点值按量化参数转换为 int8。
static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale) {
    float dst_val = (f32 / scale) + zp;  // 按量化参数换算后的整数值。
    int8_t res = (int8_t)__clip(dst_val, -128, 127);  // 裁剪到 int8 可表示范围。
    return res;
}

// 将浮点值按量化参数转换为 uint8。
static uint8_t qnt_f32_to_affine_u8(float f32, int32_t zp, float scale) {
    float dst_val = (f32 / scale) + zp;  // 按量化参数换算后的整数值。
    uint8_t res = (uint8_t)__clip(dst_val, 0, 255);  // 裁剪到 uint8 可表示范围。
    return res;
}

// 将 int8 量化值还原成浮点值。
static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) {
    return ((float)qnt - (float)zp) * scale;
}

// 将 uint8 量化值还原成浮点值。
static float deqnt_affine_u8_to_f32(uint8_t qnt, int32_t zp, float scale) {
    return ((float)qnt - (float)zp) * scale;
}

// 解析 uint8 量化输出，提取候选框、类别和目标分数。
static int process_u8(uint8_t *input,
                      int *anchor_ptr,
                      int grid_h,
                      int grid_w,
                      int height,
                      int width,
                      int stride,
                      int class_count,
                      std::vector<float> &boxes,
                      std::vector<float> &objProbs,
                      std::vector<int> &classId,
                      float threshold,
                      int32_t zp,
                      float scale) {
    (void)height;
    (void)width;

    int validCount = 0;  // 当前尺度上通过阈值筛选的候选框数量。
    int grid_len = grid_h * grid_w;  // 单个通道的网格元素总数。
    int prop_box_size = 5 + class_count;  // 每个 anchor 对应的属性长度。
    uint8_t thres_u8 = qnt_f32_to_affine_u8(threshold, zp, scale);  // 量化后的置信度阈值。

    for (int a = 0; a < 3; a++) {
        for (int i = 0; i < grid_h; i++) {
            for (int j = 0; j < grid_w; j++) {
                uint8_t box_confidence = input[(prop_box_size * a + 4) * grid_len + i * grid_w + j];
                if (box_confidence >= thres_u8) {
                    int offset = (prop_box_size * a) * grid_len + i * grid_w + j;  // 当前 anchor 在输出张量中的起始偏移。
                    uint8_t *in_ptr = input + offset;  // 当前候选框的属性起始地址。
                    float box_x = deqnt_affine_u8_to_f32(*in_ptr, zp, scale) * 2.0f - 0.5f;
                    float box_y = deqnt_affine_u8_to_f32(in_ptr[grid_len], zp, scale) * 2.0f - 0.5f;
                    float box_w = deqnt_affine_u8_to_f32(in_ptr[2 * grid_len], zp, scale) * 2.0f;
                    float box_h = deqnt_affine_u8_to_f32(in_ptr[3 * grid_len], zp, scale) * 2.0f;
                    box_x = (box_x + j) * (float)stride;
                    box_y = (box_y + i) * (float)stride;
                    box_w = box_w * box_w * (float)anchor_ptr[a * 2];
                    box_h = box_h * box_h * (float)anchor_ptr[a * 2 + 1];
                    box_x -= (box_w / 2.0f);
                    box_y -= (box_h / 2.0f);

                    uint8_t maxClassProbs = in_ptr[5 * grid_len];  // 当前网格内最高类别分数。
                    int maxClassId = 0;  // 对应的类别 ID。
                    for (int k = 1; k < class_count; ++k) {
                        uint8_t prob = in_ptr[(5 + k) * grid_len];
                        if (prob > maxClassProbs) {
                            maxClassId = k;
                            maxClassProbs = prob;
                        }
                    }
                    if (maxClassProbs > thres_u8) {
                        objProbs.push_back(deqnt_affine_u8_to_f32(maxClassProbs, zp, scale) *
                                           deqnt_affine_u8_to_f32(box_confidence, zp, scale));
                        classId.push_back(maxClassId);
                        validCount++;
                        boxes.push_back(box_x);
                        boxes.push_back(box_y);
                        boxes.push_back(box_w);
                        boxes.push_back(box_h);
                    }
                }
            }
        }
    }
    return validCount;
}

// 解析 int8 量化输出，提取候选框、类别和目标分数。
static int process_i8(int8_t *input,
                      int *anchor_ptr,
                      int grid_h,
                      int grid_w,
                      int height,
                      int width,
                      int stride,
                      int class_count,
                      std::vector<float> &boxes,
                      std::vector<float> &objProbs,
                      std::vector<int> &classId,
                      float threshold,
                      int32_t zp,
                      float scale) {
    (void)height;
    (void)width;

    int validCount = 0;  // 当前尺度上通过阈值筛选的候选框数量。
    int grid_len = grid_h * grid_w;  // 单个通道的网格元素总数。
    int prop_box_size = 5 + class_count;  // 每个 anchor 对应的属性长度。
    int8_t thres_i8 = qnt_f32_to_affine(threshold, zp, scale);  // 量化后的置信度阈值。

    for (int a = 0; a < 3; a++) {
        for (int i = 0; i < grid_h; i++) {
            for (int j = 0; j < grid_w; j++) {
                int8_t box_confidence = input[(prop_box_size * a + 4) * grid_len + i * grid_w + j];
                if (box_confidence >= thres_i8) {
                    int offset = (prop_box_size * a) * grid_len + i * grid_w + j;  // 当前 anchor 在输出张量中的起始偏移。
                    int8_t *in_ptr = input + offset;  // 当前候选框的属性起始地址。
                    float box_x = deqnt_affine_to_f32(*in_ptr, zp, scale) * 2.0f - 0.5f;
                    float box_y = deqnt_affine_to_f32(in_ptr[grid_len], zp, scale) * 2.0f - 0.5f;
                    float box_w = deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale) * 2.0f;
                    float box_h = deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale) * 2.0f;
                    box_x = (box_x + j) * (float)stride;
                    box_y = (box_y + i) * (float)stride;
                    box_w = box_w * box_w * (float)anchor_ptr[a * 2];
                    box_h = box_h * box_h * (float)anchor_ptr[a * 2 + 1];
                    box_x -= (box_w / 2.0f);
                    box_y -= (box_h / 2.0f);

                    int8_t maxClassProbs = in_ptr[5 * grid_len];  // 当前网格内最高类别分数。
                    int maxClassId = 0;  // 对应的类别 ID。
                    for (int k = 1; k < class_count; ++k) {
                        int8_t prob = in_ptr[(5 + k) * grid_len];
                        if (prob > maxClassProbs) {
                            maxClassId = k;
                            maxClassProbs = prob;
                        }
                    }
                    if (maxClassProbs > thres_i8) {
                        objProbs.push_back(deqnt_affine_to_f32(maxClassProbs, zp, scale) *
                                           deqnt_affine_to_f32(box_confidence, zp, scale));
                        classId.push_back(maxClassId);
                        validCount++;
                        boxes.push_back(box_x);
                        boxes.push_back(box_y);
                        boxes.push_back(box_w);
                        boxes.push_back(box_h);
                    }
                }
            }
        }
    }
    return validCount;
}

// 解析 RV1106 专用布局的 int8 输出，按网格顺序恢复检测框信息。
static int process_i8_rv1106(int8_t *input,
                             int *anchor_ptr,
                             int grid_h,
                             int grid_w,
                             int height,
                             int width,
                             int stride,
                             int class_count,
                             std::vector<float> &boxes,
                             std::vector<float> &boxScores,
                             std::vector<int> &classId,
                             float threshold,
                             int32_t zp,
                             float scale) {
    (void)height;
    (void)width;

    int validCount = 0;  // 当前尺度上通过阈值筛选的候选框数量。
    int8_t thres_i8 = qnt_f32_to_affine(threshold, zp, scale);  // 量化后的置信度阈值。

    int anchor_per_branch = 3;  // 每个检测头固定对应 3 个 anchor。
    int prop_box_size = 5 + class_count;  // 每个 anchor 的属性长度。
    int align_c = prop_box_size * anchor_per_branch;  // 单个网格位置在通道方向上的总长度。

    for (int h = 0; h < grid_h; h++) {
        for (int w = 0; w < grid_w; w++) {
            for (int a = 0; a < anchor_per_branch; a++) {
                int hw_offset = h * grid_w * align_c + w * align_c + a * prop_box_size;  // 当前网格和 anchor 的起始偏移。
                int8_t *hw_ptr = input + hw_offset;  // 当前候选框属性地址。
                int8_t box_confidence = hw_ptr[4];  // 目标存在置信度。

                if (box_confidence >= thres_i8) {
                    int8_t maxClassProbs = hw_ptr[5];  // 当前网格内最高类别分数。
                    int maxClassId = 0;  // 对应的类别 ID。
                    for (int k = 1; k < class_count; ++k) {
                        int8_t prob = hw_ptr[5 + k];
                        if (prob > maxClassProbs) {
                            maxClassId = k;
                            maxClassProbs = prob;
                        }
                    }

                    float box_conf_f32 = deqnt_affine_to_f32(box_confidence, zp, scale);  // 反量化后的目标置信度。
                    float class_prob_f32 = deqnt_affine_to_f32(maxClassProbs, zp, scale);  // 反量化后的类别分数。
                    float limit_score = box_conf_f32 * class_prob_f32;  // 最终用于筛选的联合分数。

                    if (limit_score > threshold) {
                        float box_x;  // 解码后的中心点 x。
                        float box_y;  // 解码后的中心点 y。
                        float box_w;  // 解码后的宽度。
                        float box_h;  // 解码后的高度。

                        box_x = deqnt_affine_to_f32(hw_ptr[0], zp, scale) * 2.0f - 0.5f;
                        box_y = deqnt_affine_to_f32(hw_ptr[1], zp, scale) * 2.0f - 0.5f;
                        box_w = deqnt_affine_to_f32(hw_ptr[2], zp, scale) * 2.0f;
                        box_h = deqnt_affine_to_f32(hw_ptr[3], zp, scale) * 2.0f;
                        box_w = box_w * box_w;
                        box_h = box_h * box_h;

                        box_x = (box_x + w) * (float)stride;
                        box_y = (box_y + h) * (float)stride;
                        box_w *= (float)anchor_ptr[a * 2];
                        box_h *= (float)anchor_ptr[a * 2 + 1];

                        box_x -= (box_w / 2.0f);
                        box_y -= (box_h / 2.0f);

                        boxes.push_back(box_x);
                        boxes.push_back(box_y);
                        boxes.push_back(box_w);
                        boxes.push_back(box_h);
                        boxScores.push_back(limit_score);
                        classId.push_back(maxClassId);
                        validCount++;
                    }
                }
            }
        }
    }
    return validCount;
}

// 解析浮点输出，提取候选框、类别和目标分数。
static int process_fp32(float *input,
                        int *anchor_ptr,
                        int grid_h,
                        int grid_w,
                        int height,
                        int width,
                        int stride,
                        int class_count,
                        std::vector<float> &boxes,
                        std::vector<float> &objProbs,
                        std::vector<int> &classId,
                        float threshold) {
    (void)height;
    (void)width;

    int validCount = 0;  // 当前尺度上通过阈值筛选的候选框数量。
    int grid_len = grid_h * grid_w;  // 单个通道的网格元素总数。
    int prop_box_size = 5 + class_count;  // 每个 anchor 对应的属性长度。

    for (int a = 0; a < 3; a++) {
        for (int i = 0; i < grid_h; i++) {
            for (int j = 0; j < grid_w; j++) {
                float box_confidence = input[(prop_box_size * a + 4) * grid_len + i * grid_w + j];
                if (box_confidence >= threshold) {
                    int offset = (prop_box_size * a) * grid_len + i * grid_w + j;  // 当前 anchor 在输出张量中的起始偏移。
                    float *in_ptr = input + offset;  // 当前候选框的属性起始地址。
                    float box_x = *in_ptr * 2.0f - 0.5f;
                    float box_y = in_ptr[grid_len] * 2.0f - 0.5f;
                    float box_w = in_ptr[2 * grid_len] * 2.0f;
                    float box_h = in_ptr[3 * grid_len] * 2.0f;
                    box_x = (box_x + j) * (float)stride;
                    box_y = (box_y + i) * (float)stride;
                    box_w = box_w * box_w * (float)anchor_ptr[a * 2];
                    box_h = box_h * box_h * (float)anchor_ptr[a * 2 + 1];
                    box_x -= (box_w / 2.0f);
                    box_y -= (box_h / 2.0f);

                    float maxClassProbs = in_ptr[5 * grid_len];  // 当前网格内最高类别分数。
                    int maxClassId = 0;  // 对应的类别 ID。
                    for (int k = 1; k < class_count; ++k) {
                        float prob = in_ptr[(5 + k) * grid_len];
                        if (prob > maxClassProbs) {
                            maxClassId = k;
                            maxClassProbs = prob;
                        }
                    }
                    if (maxClassProbs > threshold) {
                        objProbs.push_back(maxClassProbs * box_confidence);
                        classId.push_back(maxClassId);
                        validCount++;
                        boxes.push_back(box_x);
                        boxes.push_back(box_y);
                        boxes.push_back(box_w);
                        boxes.push_back(box_h);
                    }
                }
            }
        }
    }
    return validCount;
}

// 对模型原始输出执行完整后处理，生成最终检测结果列表。
int post_process(rknn_app_context_t *app_ctx,
                 void *outputs,
                 letterbox_t *letter_box,
                 float conf_threshold,
                 float nms_threshold,
                 object_detect_result_list *od_results) {
    (void)conf_threshold;

    rknn_output *_outputs = (rknn_output *)outputs;  // 统一转换后的 RKNN 输出数组。

    /*
     * 这些临时容器会在每次后处理时被频繁使用。
     * 使用 `thread_local` 可以让每个推理线程复用自己的缓存，
     * 避免高 FPS 场景里反复分配释放。
     */
    thread_local std::vector<float> filterBoxes;  // 候选框坐标缓存，按 `[x, y, w, h]` 顺序存储。
    thread_local std::vector<float> objProbs;  // 候选框联合分数缓存。
    thread_local std::vector<int> classId;  // 候选框类别 ID 缓存。
    thread_local std::vector<int> indexArray;  // 排序后的候选框索引数组。

    int validCount = 0;  // 三个尺度累计保留下来的候选框数量。
    int stride = 0;  // 当前输出尺度对应的步长。
    int grid_h = 0;  // 当前输出尺度的网格高度。
    int grid_w = 0;  // 当前输出尺度的网格宽度。
    int model_in_w = app_ctx->model_width;  // 模型输入宽度。
    int model_in_h = app_ctx->model_height;  // 模型输入高度。
    int class_count = app_ctx->class_count;  // 当前模型类别数。

    if (class_count <= 0) {
        printf("invalid yolov5 class_count=%d\n", class_count);
        return -1;
    }

    memset(od_results, 0, sizeof(object_detect_result_list));

    // `clear()` 后仍会保留 capacity，便于下一帧继续复用。
    filterBoxes.clear();
    objProbs.clear();
    classId.clear();
    indexArray.clear();
    filterBoxes.reserve(OBJ_NUMB_MAX_SIZE * 4);
    objProbs.reserve(OBJ_NUMB_MAX_SIZE);
    classId.reserve(OBJ_NUMB_MAX_SIZE);
    indexArray.reserve(OBJ_NUMB_MAX_SIZE);

    for (int i = 0; i < 3; i++) {
        grid_h = app_ctx->output_attrs[i].dims[2];
        grid_w = app_ctx->output_attrs[i].dims[3];
        stride = model_in_h / grid_h;
        if (app_ctx->is_quant) {
            validCount += process_i8((int8_t *)_outputs[i].buf,
                                     (int *)anchor[i],
                                     grid_h,
                                     grid_w,
                                     model_in_h,
                                     model_in_w,
                                     stride,
                                     class_count,
                                     filterBoxes,
                                     objProbs,
                                     classId,
                                     conf_threshold,
                                     app_ctx->output_attrs[i].zp,
                                     app_ctx->output_attrs[i].scale);
        } else {
            validCount += process_fp32((float *)_outputs[i].buf,
                                       (int *)anchor[i],
                                       grid_h,
                                       grid_w,
                                       model_in_h,
                                       model_in_w,
                                       stride,
                                       class_count,
                                       filterBoxes,
                                       objProbs,
                                       classId,
                                       conf_threshold);
        }
    }

    if (validCount <= 0) {
        return 0;
    }

    for (int i = 0; i < validCount; ++i) {
        indexArray.push_back(i);
    }
    quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

    // 标记当前帧里实际出现过的类别，只对这些类别执行 NMS。
    std::vector<bool> class_seen(static_cast<size_t>(class_count), false);
    for (size_t i = 0; i < classId.size(); ++i) {
        int cls = classId[i];
        if (cls >= 0 && cls < class_count) {
            class_seen[static_cast<size_t>(cls)] = true;
        }
    }

    for (int c = 0; c < class_count; ++c) {
        if (class_seen[static_cast<size_t>(c)]) {
            nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
        }
    }

    int last_count = 0;  // 最终写入输出结果数组的目标数。
    od_results->count = 0;

    for (int i = 0; i < validCount; ++i) {
        if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE) {
            continue;
        }
        int n = indexArray[i];  // 当前保留下来的候选框原始索引。

        float x1 = filterBoxes[n * 4 + 0] - letter_box->x_pad;  // 映射回原图前的左上角 x。
        float y1 = filterBoxes[n * 4 + 1] - letter_box->y_pad;  // 映射回原图前的左上角 y。
        float x2 = x1 + filterBoxes[n * 4 + 2];  // 映射回原图前的右下角 x。
        float y2 = y1 + filterBoxes[n * 4 + 3];  // 映射回原图前的右下角 y。
        int id = classId[n];  // 当前目标类别 ID。
        float obj_conf = objProbs[i];  // 当前目标联合分数。

        od_results->results[last_count].box.left = (int)(clamp(x1, 0, model_in_w) / letter_box->scale);
        od_results->results[last_count].box.top = (int)(clamp(y1, 0, model_in_h) / letter_box->scale);
        od_results->results[last_count].box.right = (int)(clamp(x2, 0, model_in_w) / letter_box->scale);
        od_results->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) / letter_box->scale);
        od_results->results[last_count].prop = obj_conf;
        od_results->results[last_count].cls_id = id;
        last_count++;
    }
    od_results->count = last_count;
    return 0;
}

// 初始化后处理模块，加载标签文件并缓存类别名称。
int init_post_process(const char *label_path, int expected_class_count) {
    std::vector<char *> loaded_labels;  // 从标签文件临时读出的类别名称表。

    deinit_post_process();

    if (label_path == NULL || label_path[0] == '\0') {
        printf("label path is required, please pass --labels <path>\n");
        return -1;
    }

    int ret = loadLabelName(label_path, &loaded_labels);  // 标签文件加载结果。
    if (ret < 0) {
        printf("Load %s failed!\n", label_path);
        return -1;
    }

    labels = loaded_labels;
    if (expected_class_count > 0 && !labels.empty() &&
        static_cast<int>(labels.size()) != expected_class_count) {
        printf("warning: label count=%zu does not match model class_count=%d\n",
               labels.size(),
               expected_class_count);
    }
    return 0;
}

// 根据类别 ID 返回对应的标签名称，越界时返回兜底名称。
const char *coco_cls_to_name(int cls_id) {
    if (cls_id < 0 || static_cast<size_t>(cls_id) >= labels.size()) {
        static thread_local char fallback_name[32];  // 越界类别 ID 的兜底名称缓冲。
        snprintf(fallback_name, sizeof(fallback_name), "class_%d", cls_id);
        return fallback_name;
    }

    if (labels[cls_id] != NULL) {
        return labels[cls_id];
    }

    return "null";
}

// 释放标签表占用的动态内存，清理后处理模块状态。
void deinit_post_process() {
    for (size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] != NULL) {
            free(labels[i]);
            labels[i] = NULL;
        }
    }
    labels.clear();
}
