#ifndef _RKNN_MODEL_ZOO_COMMON_H_
#define _RKNN_MODEL_ZOO_COMMON_H_

// 文件说明：
// 定义项目中通用的图像格式、图像缓冲区和矩形等基础数据结构。

// 图像像素格式枚举。
typedef enum {
    // 单通道灰度图。
    IMAGE_FORMAT_GRAY8,

    // RGB 三通道图。
    IMAGE_FORMAT_RGB888,

    // BGR 三通道图，OpenCV 中最常见。
    IMAGE_FORMAT_BGR888,

    // RGBA 四通道图。
    IMAGE_FORMAT_RGBA8888,

    // NV21 格式 YUV420SP。
    IMAGE_FORMAT_YUV420SP_NV21,

    // NV12 格式 YUV420SP。
    IMAGE_FORMAT_YUV420SP_NV12,
} image_format_t;

// 通用图像缓冲区描述。
typedef struct {
    // 图像逻辑宽度。
    int width;

    // 图像逻辑高度。
    int height;

    // 图像宽方向步长。
    int width_stride;

    // 图像高方向步长。
    int height_stride;

    // 图像像素格式。
    image_format_t format;

    // 用户态可访问的虚拟地址。
    unsigned char *virt_addr;

    // 缓冲区总字节数。
    int size;

    // 底层缓冲区对应的文件描述符，某些硬件路径会使用。
    int fd;
} image_buffer_t;

// 轴对齐矩形框。
typedef struct {
    // 左边界。
    int left;

    // 上边界。
    int top;

    // 右边界。
    int right;

    // 下边界。
    int bottom;
} image_rect_t;

// 旋转矩形框。
typedef struct {
    // 左上角 x。
    int x;

    // 左上角 y。
    int y;

    // 宽度。
    int w;

    // 高度。
    int h;

    // 旋转角度。
    float angle;
} image_obb_box_t;

#endif //_RKNN_MODEL_ZOO_COMMON_H_
