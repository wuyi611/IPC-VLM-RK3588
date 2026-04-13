#include "model/rknn_input_buffer_utils.h"

#include <stdio.h>
#include <string.h>

#include "dma_alloc.h"
#include "image_utils.h"

// 文件说明：
// 统一实现 RKNN 输入 DMA-BUF 的申请、绑定和释放。

namespace {

// 按优先级尝试的 DMA heap 路径列表。
const char *kDmaHeapCandidates[] = {
    DMA_HEAP_UNCACHE_PATH,
    CMA_HEAP_UNCACHE_PATH,
    DMA_HEAP_PATH,
};

}  // namespace

// 为 RKNN 输入 tensor 创建一块长期复用的 DMA-BUF。
int InitRknnInputBuffer(rknn_app_context_t *app_ctx) {
    if (app_ctx == NULL || app_ctx->input_attrs == NULL) {
        return -1;
    }

    memset(&app_ctx->model_input_buffer, 0, sizeof(app_ctx->model_input_buffer));
    app_ctx->model_input_buffer.width = app_ctx->model_width;
    app_ctx->model_input_buffer.height = app_ctx->model_height;
    app_ctx->model_input_buffer.width_stride =
        app_ctx->input_attrs[0].w_stride > 0
            ? static_cast<int>(app_ctx->input_attrs[0].w_stride)
            : app_ctx->model_width;
    app_ctx->model_input_buffer.height_stride = app_ctx->model_height;
    app_ctx->model_input_buffer.format = IMAGE_FORMAT_RGB888;
    app_ctx->model_input_buffer.fd = -1;

    // 真实需要申请的输入缓冲大小，优先使用 RKNN 给出的 stride 后尺寸。
    uint32_t buffer_size =
        app_ctx->input_attrs[0].size_with_stride > 0
            ? app_ctx->input_attrs[0].size_with_stride
            : static_cast<uint32_t>(get_image_size(&app_ctx->model_input_buffer));
    app_ctx->model_input_buffer.size = static_cast<int>(buffer_size);

    int ret = -1;  // 记录 DMA 申请或 RKNN 接口的返回值。
    for (size_t i = 0; i < sizeof(kDmaHeapCandidates) / sizeof(kDmaHeapCandidates[0]); ++i) {
        ret = dma_buf_alloc(
            kDmaHeapCandidates[i],
            static_cast<size_t>(buffer_size),
            &app_ctx->model_input_buffer.fd,
            reinterpret_cast<void **>(&app_ctx->model_input_buffer.virt_addr));
        if (ret == 0) {
            break;
        }
    }
    if (ret != 0 || app_ctx->model_input_buffer.fd < 0) {
        printf("dma_buf_alloc fail! size=%u ret=%d\n", buffer_size, ret);
        return -1;
    }

    // 把 DMA-BUF 包装成 RKNN 可识别的外部输入内存对象。
    app_ctx->input_tensor_mem = rknn_create_mem_from_fd(app_ctx->rknn_ctx,
                                                        app_ctx->model_input_buffer.fd,
                                                        app_ctx->model_input_buffer.virt_addr,
                                                        buffer_size,
                                                        0);
    if (app_ctx->input_tensor_mem == NULL) {
        printf("rknn_create_mem_from_fd fail!\n");
        ReleaseRknnInputBuffer(app_ctx);
        return -1;
    }

    // 构造送给 `rknn_set_io_mem` 的输入属性副本。
    memset(&app_ctx->input_io_attr, 0, sizeof(app_ctx->input_io_attr));
    app_ctx->input_io_attr = app_ctx->input_attrs[0];
    app_ctx->input_io_attr.index = 0;
    app_ctx->input_io_attr.type = RKNN_TENSOR_UINT8;
    app_ctx->input_io_attr.fmt = RKNN_TENSOR_NHWC;
    app_ctx->input_io_attr.pass_through = 0;
    app_ctx->input_io_attr.h_stride =
        static_cast<uint32_t>(app_ctx->model_input_buffer.height_stride);

    ret = rknn_set_io_mem(app_ctx->rknn_ctx, app_ctx->input_tensor_mem, &app_ctx->input_io_attr);
    if (ret != RKNN_SUCC) {
        // 某些平台对传入属性更敏感，这里退回到原始输入属性再试一次。
        rknn_tensor_attr fallback_attr = app_ctx->input_attrs[0];
        fallback_attr.index = 0;
        fallback_attr.pass_through = 0;
        fallback_attr.h_stride =
            static_cast<uint32_t>(app_ctx->model_input_buffer.height_stride);
        ret = rknn_set_io_mem(app_ctx->rknn_ctx, app_ctx->input_tensor_mem, &fallback_attr);
        if (ret != RKNN_SUCC) {
            printf("rknn_set_io_mem fail! ret=%d\n", ret);
            ReleaseRknnInputBuffer(app_ctx);
            return -1;
        }
        app_ctx->input_io_attr = fallback_attr;
    }

    return 0;
}

// 释放 `InitRknnInputBuffer` 申请的 DMA-BUF 和 RKNN 外部内存对象。
void ReleaseRknnInputBuffer(rknn_app_context_t *app_ctx) {
    if (app_ctx == NULL) {
        return;
    }

    if (app_ctx->input_tensor_mem != NULL) {
        rknn_destroy_mem(app_ctx->rknn_ctx, app_ctx->input_tensor_mem);
        app_ctx->input_tensor_mem = NULL;
    }

    if (app_ctx->model_input_buffer.fd >= 0) {
        dma_buf_free(static_cast<size_t>(app_ctx->model_input_buffer.size),
                     &app_ctx->model_input_buffer.fd,
                     app_ctx->model_input_buffer.virt_addr);
    }

    app_ctx->model_input_buffer.virt_addr = NULL;
    app_ctx->model_input_buffer.size = 0;
    app_ctx->model_input_buffer.width = 0;
    app_ctx->model_input_buffer.height = 0;
    app_ctx->model_input_buffer.width_stride = 0;
    app_ctx->model_input_buffer.height_stride = 0;
    memset(&app_ctx->input_io_attr, 0, sizeof(app_ctx->input_io_attr));
}
