#ifndef RKNN_YOLOV5_DEMO_MODEL_POSTPROCESS_H_
#define RKNN_YOLOV5_DEMO_MODEL_POSTPROCESS_H_

#include <cstddef>
#include <stdint.h>

#include <vector>

#include "common.h"
#include "image_utils.h"
#include "rknn_api.h"

// 文件说明：
// 声明 YOLOv5 输出张量的后处理结构、阈值常量和后处理接口。

// 类别名最大长度。
#define OBJ_NAME_MAX_SIZE 64

// 单帧最多保留的检测目标数量。
#define OBJ_NUMB_MAX_SIZE 128

// 默认 NMS 阈值。
#define NMS_THRESH 0.45

// 默认置信度阈值。
#define BOX_THRESH 0.25

struct rknn_app_context_t;

// 表示单个检测结果。
typedef struct {
    // 目标框坐标。
    image_rect_t box;

    // 目标置信度。
    float prop;

    // 类别 ID。
    int cls_id;
} object_detect_result;

// 表示单帧检测结果列表。
typedef struct {
    // 结果集 ID，当前实现中通常未额外使用。
    int id;

    // 当前有效检测结果数量。
    int count;

    // 检测结果数组。
    object_detect_result results[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;

// 初始化后处理模块，例如加载标签文件。
// `label_path` 必须由调用方显式提供。
// `expected_class_count > 0` 时会校验标签数量是否和模型类别数一致。
int init_post_process(const char *label_path, int expected_class_count = 0);

// 释放后处理模块资源。
void deinit_post_process();

// 根据类别 ID 返回当前标签表中的类别名称。
const char *coco_cls_to_name(int cls_id);

// 对模型输出做后处理，生成最终检测框列表。
int post_process(rknn_app_context_t *app_ctx,
                 void *outputs,
                 letterbox_t *letter_box,
                 float conf_threshold,
                 float nms_threshold,
                 object_detect_result_list *od_results);

// 旧接口别名，保留兼容性。
void deinitPostProcess();

#endif  // RKNN_YOLOV5_DEMO_MODEL_POSTPROCESS_H_
