#include "display/drm_display.h"

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <utility>

#include <drm/drm_fourcc.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

#include "dma_alloc.h"
#include "image_drawing.h"
#include "model/postprocess.h"

// 文件说明：
// 实现基于 DRM/KMS 的显示输出：
// 1. 只申请一个 Overlay Plane 做叠加显示；
// 2. 不占用系统 Primary Plane，也不主动接管整块屏幕。

namespace rknn_demo {

namespace {

const int kMaxDrmCards = 16;

// 辅助函数：在图像的指定位置绘制一行文本
void DrawTextLine(image_buffer_t *image,
                  const char *text,
                  int x,
                  int y,
                  unsigned int color,
                  int font_size) {
    if (image == NULL || text == NULL) {
        return;
    }
    draw_text(image, text, x, y, color, font_size);
}

// 辅助函数：检查指定的 DRM Plane 是否支持给定的像素格式 (FOURCC)
bool PlaneSupportsFormat(const drmModePlane *plane, uint32_t fourcc) {
    if (plane == NULL || fourcc == 0) {
        return false;
    }
    for (uint32_t i = 0; i < plane->count_formats; ++i) {
        if (plane->formats[i] == fourcc) {
            return true;
        }
    }
    return false;
}

// 辅助函数：计算图像的 Letterbox (等比例缩放并居中填黑) 布局参数
bool ComputeLetterboxLayout(const image_buffer_t &src_image,
                            int dst_w,
                            int dst_h,
                            letterbox_t *letter_box) {
    if (src_image.width <= 0 || src_image.height <= 0 || dst_w <= 0 || dst_h <= 0) {
        return false;
    }

    float scale_w = static_cast<float>(dst_w) / static_cast<float>(src_image.width);
    float scale_h = static_cast<float>(dst_h) / static_cast<float>(src_image.height);
    float scale = scale_w < scale_h ? scale_w : scale_h;

    int resize_w = scale_w < scale_h
                       ? dst_w
                       : static_cast<int>(src_image.width * scale);
    int resize_h = scale_w < scale_h
                       ? static_cast<int>(src_image.height * scale)
                       : dst_h;

    if (resize_w % 4 != 0) {
        resize_w -= resize_w % 4;
    }
    if (resize_h % 2 != 0) {
        resize_h -= resize_h % 2;
    }
    if (resize_w <= 0 || resize_h <= 0) {
        return false;
    }

    int left = 0;
    int top = 0;
    if (scale_w < scale_h) {
        top = (dst_h - resize_h) / 2;
        if (top % 2 != 0) {
            top -= top % 2;
        }
        if (top < 0) {
            top = 0;
        }
    } else {
        left = (dst_w - resize_w) / 2;
        if (left % 2 != 0) {
            left -= left % 2;
        }
        if (left < 0) {
            left = 0;
        }
    }

    if (letter_box != NULL) {
        letter_box->scale = scale;
        letter_box->x_pad = left;
        letter_box->y_pad = top;
    }
    return true;
}

// 辅助函数：获取单调递增的时间，单位为毫秒，常用于计算耗时或超时
double MonotonicMs() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<double>(ts.tv_sec) * 1000.0 +
           static_cast<double>(ts.tv_nsec) / 1000000.0;
}

}  // namespace

// DrmDisplay 构造函数：初始化内部状态变量
DrmDisplay::DrmDisplay()
    : drm_fd_(-1),
      connector_id_(0),
      crtc_id_(0),
      crtc_index_(-1),
      saved_crtc_(NULL),
      front_buffer_index_(0),
      pending_buffer_index_(-1),
      page_flip_pending_(false),
      initialized_(false),
      overlay_started_(false),
      restore_state_captured_(false),
      saved_connector_crtc_id_value_(0) {
    memset(&mode_, 0, sizeof(mode_));
}

// DrmDisplay 析构函数：确保释放 DRM 资源并恢复显示状态
DrmDisplay::~DrmDisplay() {
    Close();
}

// 初始化 DRM 显示子系统：包括打开设备、选择连接器、分配缓冲区以及设置 Overlay
bool DrmDisplay::Init() {
    Close();

    if (!OpenDevice()) {
        return false;
    }
    if (!SelectConnectorAndCrtc()) {
        Close();
        return false;
    }
    if (!CreateScanoutBuffers()) {
        Close();
        return false;
    }
    CaptureRestoreState();

    if (InitOverlayPlanes()) {
        initialized_ = true;
        printf("drm_display: using overlay plane only on %s\n",
               drm_card_path_.c_str());
        return true;
    }
    Close();
    return false;
}

// 渲染并呈现一帧图像及检测结果（对外接口）
bool DrmDisplay::Present(MppFrameHandle frame,
                         const object_detect_result_list &detections,
                         double render_fps,
                         double latency_ms) {
    if (!initialized_ || frame.empty()) {
        return false;
    }
    return PresentWithOverlay(std::move(frame), detections, render_fps, latency_ms);
}

// 关闭 DRM 设备，释放所有分配的 dumb buffer，并尝试恢复桌面原有状态
void DrmDisplay::Close() {
    initialized_ = false;

    if (drm_fd_ >= 0) {
        if (!WaitForPendingFlip(1000)) {
            printf("drm_display: timeout waiting pending flip before restore, continue cleanup\n");
        }
        if (!RestoreDisplayState()) {
            printf("drm_display: restore display state failed, only releasing overlay plane and DRM master\n");
        }
    }
    if (saved_crtc_ != NULL) {
        drmModeFreeCrtc(saved_crtc_);
        saved_crtc_ = NULL;
    }

    ResetOverlayPlaneState();

    for (int i = 0; i < kBufferCount; ++i) {
        DisplayBuffer &buffer = buffers_[i];
        if (drm_fd_ >= 0 && buffer.fb_id != 0) {
            drmModeRmFB(drm_fd_, buffer.fb_id);
            buffer.fb_id = 0;
        }
        if (buffer.map != NULL && buffer.size > 0) {
            munmap(buffer.map, buffer.size);
            buffer.map = NULL;
        }
        if (buffer.prime_fd >= 0) {
            close(buffer.prime_fd);
            buffer.prime_fd = -1;
        }
        if (drm_fd_ >= 0 && buffer.handle != 0) {
            struct drm_mode_destroy_dumb destroy_arg;
            memset(&destroy_arg, 0, sizeof(destroy_arg));
            destroy_arg.handle = buffer.handle;
            ioctl(drm_fd_, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy_arg);
            buffer.handle = 0;
        }

        memset(&buffer.rga_image, 0, sizeof(buffer.rga_image));
        memset(&buffer.cpu_image, 0, sizeof(buffer.cpu_image));
        buffer.rga_image.fd = -1;
        buffer.cpu_image.fd = -1;
        buffer.pitch = 0;
        buffer.size = 0;
    }

    if (drm_fd_ >= 0) {
        drmDropMaster(drm_fd_);
        close(drm_fd_);
        drm_fd_ = -1;
    }

    drm_card_path_.clear();
    connector_id_ = 0;
    crtc_id_ = 0;
    crtc_index_ = -1;
    front_buffer_index_ = 0;
    pending_buffer_index_ = -1;
    page_flip_pending_ = false;
    restore_state_captured_ = false;
    saved_connector_crtc_id_value_ = 0;
    saved_plane_states_.clear();
    memset(&mode_, 0, sizeof(mode_));
}

// 遍历系统中的 DRM 显卡设备节点 (/dev/dri/cardX)，寻找可用的 KMS 设备
bool DrmDisplay::OpenDevice() {
    for (int i = 0; i < kMaxDrmCards; ++i) {
        char path[32];
        snprintf(path, sizeof(path), "/dev/dri/card%d", i);
        if (OpenDeviceAtPath(path)) {
            return true;
        }
    }
    printf("drm_display: no usable /dev/dri/card* device found\n");
    return false;
}

// 尝试打开指定的 DRM 设备路径，并验证其是否具有 KMS 能力和有效连接
bool DrmDisplay::OpenDeviceAtPath(const char *path) {
    int fd = open(path, O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        return false;
    }
    if (drmIsKMS(fd) != 1) {
        close(fd);
        return false;
    }

    drmModeRes *resources = drmModeGetResources(fd);
    if (resources == NULL || resources->count_connectors <= 0) {
        if (resources != NULL) {
            drmModeFreeResources(resources);
        }
        close(fd);
        return false;
    }

    bool has_connected_connector = false;
    for (int i = 0; i < resources->count_connectors; ++i) {
        drmModeConnector *connector = drmModeGetConnector(fd, resources->connectors[i]);
        if (connector == NULL) {
            continue;
        }
        if (connector->connection == DRM_MODE_CONNECTED && connector->count_modes > 0) {
            has_connected_connector = true;
            drmModeFreeConnector(connector);
            break;
        }
        drmModeFreeConnector(connector);
    }
    drmModeFreeResources(resources);
    if (!has_connected_connector) {
        close(fd);
        return false;
    }

    drm_fd_ = fd;
    drm_card_path_ = path;
    return true;
}

// 在打开的 DRM 设备上寻找已连接的显示器 (Connector)，并解析对应的 Encoder 和 CRTC
bool DrmDisplay::SelectConnectorAndCrtc() {
    drmModeRes *resources = drmModeGetResources(drm_fd_);
    if (resources == NULL) {
        printf("drm_display: drmModeGetResources failed\n");
        return false;
    }

    drmModeConnector *selected_connector = NULL;
    drmModeEncoder *selected_encoder = NULL;

    for (int i = 0; i < resources->count_connectors; ++i) {
        drmModeConnector *connector = drmModeGetConnector(drm_fd_, resources->connectors[i]);
        if (connector == NULL) {
            continue;
        }
        if (connector->connection == DRM_MODE_CONNECTED && connector->count_modes > 0) {
            selected_connector = connector;
            break;
        }
        drmModeFreeConnector(connector);
    }

    if (selected_connector == NULL) {
        drmModeFreeResources(resources);
        printf("drm_display: no connected connector found on %s\n", drm_card_path_.c_str());
        return false;
    }

    connector_id_ = selected_connector->connector_id;
    mode_ = selected_connector->modes[0];
    for (int i = 0; i < selected_connector->count_modes; ++i) {
        if (selected_connector->modes[i].type & DRM_MODE_TYPE_PREFERRED) {
            mode_ = selected_connector->modes[i];
            break;
        }
    }

    if (selected_connector->encoder_id != 0) {
        selected_encoder = drmModeGetEncoder(drm_fd_, selected_connector->encoder_id);
    }
    if (selected_encoder == NULL) {
        for (int i = 0; i < selected_connector->count_encoders; ++i) {
            selected_encoder = drmModeGetEncoder(drm_fd_, selected_connector->encoders[i]);
            if (selected_encoder != NULL) {
                break;
            }
        }
    }

    if (selected_encoder == NULL) {
        drmModeFreeConnector(selected_connector);
        drmModeFreeResources(resources);
        printf("drm_display: no encoder found for connector %u\n", connector_id_);
        return false;
    }

    crtc_id_ = 0;
    crtc_index_ = -1;
    if (selected_encoder->crtc_id != 0) {
        for (int i = 0; i < resources->count_crtcs; ++i) {
            if (resources->crtcs[i] == selected_encoder->crtc_id) {
                crtc_id_ = selected_encoder->crtc_id;
                crtc_index_ = i;
                break;
            }
        }
    }
    if (crtc_id_ == 0 || crtc_index_ < 0) {
        for (int i = 0; i < resources->count_crtcs; ++i) {
            if (selected_encoder->possible_crtcs & (1 << i)) {
                crtc_id_ = resources->crtcs[i];
                crtc_index_ = i;
                break;
            }
        }
    }

    saved_crtc_ = drmModeGetCrtc(drm_fd_, crtc_id_);

    drmModeFreeEncoder(selected_encoder);
    drmModeFreeConnector(selected_connector);
    drmModeFreeResources(resources);

    if (crtc_id_ == 0 || crtc_index_ < 0) {
        printf("drm_display: failed to resolve crtc\n");
        return false;
    }
    return true;
}

// 分配 DRM Dumb Buffer 并将其映射到用户空间内存，同时获取 dma-buf fd 供硬件加速模块(如RGA)使用
bool DrmDisplay::CreateScanoutBuffers() {
    for (int i = 0; i < kBufferCount; ++i) {
        DisplayBuffer &buffer = buffers_[i];
        struct drm_mode_create_dumb create_arg;
        memset(&create_arg, 0, sizeof(create_arg));
        create_arg.width = static_cast<uint32_t>(mode_.hdisplay);
        create_arg.height = static_cast<uint32_t>(mode_.vdisplay);
        create_arg.bpp = 32;

        if (ioctl(drm_fd_, DRM_IOCTL_MODE_CREATE_DUMB, &create_arg) != 0) {
            printf("drm_display: DRM_IOCTL_MODE_CREATE_DUMB failed\n");
            return false;
        }

        buffer.handle = create_arg.handle;
        buffer.pitch = create_arg.pitch;
        buffer.size = static_cast<size_t>(create_arg.size);

        struct drm_mode_map_dumb map_arg;
        memset(&map_arg, 0, sizeof(map_arg));
        map_arg.handle = buffer.handle;
        if (ioctl(drm_fd_, DRM_IOCTL_MODE_MAP_DUMB, &map_arg) != 0) {
            printf("drm_display: DRM_IOCTL_MODE_MAP_DUMB failed\n");
            return false;
        }

        buffer.map = mmap(NULL,
                          buffer.size,
                          PROT_READ | PROT_WRITE,
                          MAP_SHARED,
                          drm_fd_,
                          map_arg.offset);
        if (buffer.map == MAP_FAILED) {
            buffer.map = NULL;
            printf("drm_display: mmap dumb buffer failed\n");
            return false;
        }
        memset(buffer.map, 0, buffer.size);

        if (drmPrimeHandleToFD(drm_fd_, buffer.handle, DRM_CLOEXEC, &buffer.prime_fd) != 0) {
            printf("drm_display: drmPrimeHandleToFD failed\n");
            return false;
        }

        uint32_t handles[4] = {buffer.handle, 0, 0, 0};
        uint32_t pitches[4] = {buffer.pitch, 0, 0, 0};
        uint32_t offsets[4] = {0, 0, 0, 0};
        if (drmModeAddFB2(drm_fd_,
                          static_cast<uint32_t>(mode_.hdisplay),
                          static_cast<uint32_t>(mode_.vdisplay),
                          DRM_FORMAT_ABGR8888,
                          handles,
                          pitches,
                          offsets,
                          &buffer.fb_id,
                          0) != 0) {
            printf("drm_display: drmModeAddFB2 failed\n");
            return false;
        }

        memset(&buffer.rga_image, 0, sizeof(buffer.rga_image));
        buffer.rga_image.width = mode_.hdisplay;
        buffer.rga_image.height = mode_.vdisplay;
        buffer.rga_image.width_stride = static_cast<int>(buffer.pitch / 4);
        buffer.rga_image.height_stride = mode_.vdisplay;
        buffer.rga_image.format = IMAGE_FORMAT_RGBA8888;
        buffer.rga_image.fd = buffer.prime_fd;
        buffer.rga_image.size = static_cast<int>(buffer.size);
        buffer.rga_image.virt_addr = NULL;

        buffer.cpu_image = buffer.rga_image;
        buffer.cpu_image.virt_addr = static_cast<unsigned char *>(buffer.map);
    }
    return true;
}

// 捕获当前的 DRM 属性状态 (如 CRTC 和各类 Plane 参数)，以便在程序退出时恢复桌面原有状态
bool DrmDisplay::CaptureRestoreState() {
    restore_state_captured_ = false;
    saved_connector_crtc_id_value_ = 0;
    saved_plane_states_.clear();

    if (drmSetClientCap(drm_fd_, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1) != 0) {
        return false;
    }
    if (drmSetClientCap(drm_fd_, DRM_CLIENT_CAP_ATOMIC, 1) != 0) {
        return false;
    }

    if (!GetObjectProperty(connector_id_,
                           DRM_MODE_OBJECT_CONNECTOR,
                           "CRTC_ID",
                           NULL,
                           &saved_connector_crtc_id_value_,
                           NULL)) {
        return false;
    }

    drmModePlaneRes *plane_resources = drmModeGetPlaneResources(drm_fd_);
    if (plane_resources == NULL) {
        return false;
    }

    for (uint32_t i = 0; i < plane_resources->count_planes; ++i) {
        drmModePlane *plane = drmModeGetPlane(drm_fd_, plane_resources->planes[i]);
        if (plane == NULL) {
            continue;
        }
        if (crtc_index_ >= 0 && (plane->possible_crtcs & (1u << crtc_index_)) != 0) {
            PlaneState saved_state;
            if (LoadPlaneState(plane->plane_id, &saved_state)) {
                saved_plane_states_.push_back(saved_state);
            }
        }
        drmModeFreePlane(plane);
    }
    drmModeFreePlaneResources(plane_resources);

    restore_state_captured_ = !saved_plane_states_.empty() || saved_crtc_ != NULL;
    return restore_state_captured_;
}

// 重置 Overlay Plane 的内部状态，清理相关的配置缓存
void DrmDisplay::ResetOverlayPlaneState() {
    overlay_started_ = false;
    page_flip_pending_ = false;
    pending_buffer_index_ = -1;

    connector_props_ = ConnectorProperties();
    crtc_props_ = CrtcProperties();
    video_plane_ = PlaneState();
}

// 等待之前的 DRM Page Flip (切帧) 操作完成，支持超时设置
bool DrmDisplay::WaitForPendingFlip(int timeout_ms) {
    if (!page_flip_pending_) {
        return true;
    }

    double deadline_ms = timeout_ms > 0 ? static_cast<double>(timeout_ms) : 0.0;
    double start_ms = MonotonicMs();
    while (page_flip_pending_) {
        int wait_ms = 20;
        if (timeout_ms > 0) {
            double elapsed_ms = MonotonicMs() - start_ms;
            if (elapsed_ms >= deadline_ms) {
                break;
            }
            wait_ms = static_cast<int>(deadline_ms - elapsed_ms);
            if (wait_ms > 20) {
                wait_ms = 20;
            }
            if (wait_ms <= 0) {
                wait_ms = 1;
            }
        }
        if (!PumpEvents(wait_ms)) {
            return false;
        }
    }
    return !page_flip_pending_;
}

// 执行 DRM Atomic 提交，将显卡状态还原到 CaptureRestoreState 时捕获的初始桌面状态
bool DrmDisplay::RestoreDisplayState() {
    if (drm_fd_ < 0) {
        return false;
    }

    bool restored = false;
    if (restore_state_captured_ && saved_crtc_ != NULL) {
        drmModeAtomicReq *request = drmModeAtomicAlloc();
        if (request != NULL) {
            uint32_t restore_mode_blob_id = 0;
            bool success = drmModeAtomicAddProperty(request,
                                                    connector_id_,
                                                    connector_props_.crtc_id,
                                                    saved_connector_crtc_id_value_) >= 0;

            uint64_t restore_active = saved_crtc_->mode_valid ? 1 : 0;
            if (success && saved_crtc_->mode_valid) {
                success = drmModeCreatePropertyBlob(drm_fd_,
                                                    &saved_crtc_->mode,
                                                    sizeof(saved_crtc_->mode),
                                                    &restore_mode_blob_id) == 0;
            }
            if (success) {
                success = drmModeAtomicAddProperty(request,
                                                   crtc_id_,
                                                   crtc_props_.active,
                                                   restore_active) >= 0;
            }
            if (success) {
                success = drmModeAtomicAddProperty(request,
                                                   crtc_id_,
                                                   crtc_props_.mode_id,
                                                   saved_crtc_->mode_valid ? restore_mode_blob_id : 0) >= 0;
            }
            for (size_t i = 0; success && i < saved_plane_states_.size(); ++i) {
                success = AddRestorePlaneToAtomicRequest(request, saved_plane_states_[i]);
            }

            if (success &&
                drmModeAtomicCommit(drm_fd_,
                                    request,
                                    DRM_MODE_ATOMIC_ALLOW_MODESET,
                                    NULL) == 0) {
                restored = true;
            }

            if (restore_mode_blob_id != 0) {
                drmModeDestroyPropertyBlob(drm_fd_, restore_mode_blob_id);
            }
            drmModeAtomicFree(request);
        }
    }

    if (!restored && overlay_started_) {
        if (video_plane_.valid()) {
            drmModeSetPlane(drm_fd_, video_plane_.plane_id, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        }
    }
    return restored;
}

// 检查 DRM 是否满足 Overlay 模式需求（桌面必须处于点亮状态），并找到合适的视频叠加层
bool DrmDisplay::InitOverlayPlanes() {
    ResetOverlayPlaneState();

    if (drmSetClientCap(drm_fd_, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1) != 0) {
        return false;
    }
    if (drmSetClientCap(drm_fd_, DRM_CLIENT_CAP_ATOMIC, 1) != 0) {
        return false;
    }

    if (saved_crtc_ == NULL || !saved_crtc_->mode_valid ||
        saved_connector_crtc_id_value_ != crtc_id_) {
        printf("drm_display: overlay-only mode requires an already active primary desktop on crtc=%u\n",
               crtc_id_);
        return false;
    }

    if (!GetObjectProperty(connector_id_,
                           DRM_MODE_OBJECT_CONNECTOR,
                           "CRTC_ID",
                           &connector_props_.crtc_id,
                           NULL,
                           NULL)) {
        return false;
    }
    if (!GetObjectProperty(crtc_id_,
                           DRM_MODE_OBJECT_CRTC,
                           "ACTIVE",
                           &crtc_props_.active,
                           NULL,
                           NULL)) {
        return false;
    }
    if (!GetObjectProperty(crtc_id_,
                           DRM_MODE_OBJECT_CRTC,
                           "MODE_ID",
                           &crtc_props_.mode_id,
                           NULL,
                           NULL)) {
        return false;
    }
    if (!SelectPlaneForFormat(DRM_PLANE_TYPE_OVERLAY, DRM_FORMAT_ABGR8888, 0, &video_plane_)) {
        return false;
    }

    return true;
}

// 从当前系统中挑选出一个满足类型(Type)且支持特定格式(FOURCC)的 DRM Plane
bool DrmDisplay::SelectPlaneForFormat(uint64_t desired_type,
                                      uint32_t fourcc,
                                      uint32_t excluded_plane_id,
                                      PlaneState *plane_state) {
    drmModePlaneRes *plane_resources = drmModeGetPlaneResources(drm_fd_);
    if (plane_resources == NULL) {
        return false;
    }

    bool found = false;
    PlaneState selected_plane;
    for (uint32_t i = 0; i < plane_resources->count_planes; ++i) {
        drmModePlane *plane = drmModeGetPlane(drm_fd_, plane_resources->planes[i]);
        if (plane == NULL) {
            continue;
        }

        bool compatible = crtc_index_ >= 0 &&
                          (plane->possible_crtcs & (1u << crtc_index_)) != 0;
        bool format_supported = PlaneSupportsFormat(plane, fourcc);
        bool plane_available = plane->plane_id != excluded_plane_id;

        if (compatible && format_supported && plane_available) {
            PlaneState candidate;
            if (LoadPlaneState(plane->plane_id, &candidate) &&
                candidate.type_value == desired_type) {
                if (!found || (!selected_plane.has_alpha && candidate.has_alpha)) {
                    selected_plane = candidate;
                    found = true;
                }
            }
        }

        drmModeFreePlane(plane);
    }
    drmModeFreePlaneResources(plane_resources);

    if (found && plane_state != NULL) {
        *plane_state = selected_plane;
    }
    return found;
}

// 通过 DRM Atomic 接口查询指定 Plane 的各类属性值（坐标、FB、混合模式等）并缓存
bool DrmDisplay::LoadPlaneState(uint32_t plane_id, PlaneState *plane_state) const {
    if (plane_state == NULL) {
        return false;
    }

    PlaneState state;
    state.plane_id = plane_id;

    if (!GetObjectProperty(plane_id,
                           DRM_MODE_OBJECT_PLANE,
                           "type",
                           &state.props.type,
                           &state.type_value,
                           NULL) ||
        !GetObjectProperty(plane_id,
                           DRM_MODE_OBJECT_PLANE,
                           "CRTC_ID",
                           &state.props.crtc_id,
                           &state.current_crtc_id,
                           NULL) ||
        !GetObjectProperty(plane_id,
                           DRM_MODE_OBJECT_PLANE,
                           "FB_ID",
                           &state.props.fb_id,
                           &state.current_fb_id,
                           NULL) ||
        !GetObjectProperty(plane_id,
                           DRM_MODE_OBJECT_PLANE,
                           "CRTC_X",
                           &state.props.crtc_x,
                           &state.current_crtc_x,
                           NULL) ||
        !GetObjectProperty(plane_id,
                           DRM_MODE_OBJECT_PLANE,
                           "CRTC_Y",
                           &state.props.crtc_y,
                           &state.current_crtc_y,
                           NULL) ||
        !GetObjectProperty(plane_id,
                           DRM_MODE_OBJECT_PLANE,
                           "CRTC_W",
                           &state.props.crtc_w,
                           &state.current_crtc_w,
                           NULL) ||
        !GetObjectProperty(plane_id,
                           DRM_MODE_OBJECT_PLANE,
                           "CRTC_H",
                           &state.props.crtc_h,
                           &state.current_crtc_h,
                           NULL) ||
        !GetObjectProperty(plane_id,
                           DRM_MODE_OBJECT_PLANE,
                           "SRC_X",
                           &state.props.src_x,
                           &state.current_src_x,
                           NULL) ||
        !GetObjectProperty(plane_id,
                           DRM_MODE_OBJECT_PLANE,
                           "SRC_Y",
                           &state.props.src_y,
                           &state.current_src_y,
                           NULL) ||
        !GetObjectProperty(plane_id,
                           DRM_MODE_OBJECT_PLANE,
                           "SRC_W",
                           &state.props.src_w,
                           &state.current_src_w,
                           NULL) ||
        !GetObjectProperty(plane_id,
                           DRM_MODE_OBJECT_PLANE,
                           "SRC_H",
                           &state.props.src_h,
                           &state.current_src_h,
                           NULL)) {
        return false;
    }

    drmModePropertyRes *alpha_property = NULL;
    if (GetObjectProperty(plane_id,
                          DRM_MODE_OBJECT_PLANE,
                          "alpha",
                          &state.props.alpha,
                          &state.current_alpha,
                          &alpha_property)) {
        state.has_alpha = true;
        if (alpha_property->count_values > 0) {
            state.alpha_opaque = alpha_property->values[alpha_property->count_values - 1];
        } else {
            state.alpha_opaque = 0xffff;
        }
        drmModeFreeProperty(alpha_property);
    }

    drmModePropertyRes *blend_property = NULL;
    if (GetObjectProperty(plane_id,
                          DRM_MODE_OBJECT_PLANE,
                          "pixel blend mode",
                          &state.props.pixel_blend_mode,
                          &state.current_pixel_blend_mode,
                          &blend_property)) {
        for (int i = 0; i < blend_property->count_enums; ++i) {
            if (strcmp(blend_property->enums[i].name, "Coverage") == 0 ||
                strcmp(blend_property->enums[i].name, "Pre-multiplied") == 0) {
                state.has_pixel_blend_mode = true;
                state.blend_mode_value = blend_property->enums[i].value;
                break;
            }
        }
        drmModeFreeProperty(blend_property);
    }

    *plane_state = state;
    return true;
}

// 获取具体 DRM 对象的特定属性 (例如根据名字查找到 prop_id 和当前值)
bool DrmDisplay::GetObjectProperty(uint32_t object_id,
                                   uint32_t object_type,
                                   const char *property_name,
                                   uint32_t *property_id,
                                   uint64_t *value,
                                   drmModePropertyRes **property) const {
    drmModeObjectProperties *object_properties =
        drmModeObjectGetProperties(drm_fd_, object_id, object_type);
    if (object_properties == NULL) {
        return false;
    }

    bool found = false;
    for (uint32_t i = 0; i < object_properties->count_props; ++i) {
        drmModePropertyRes *candidate =
            drmModeGetProperty(drm_fd_, object_properties->props[i]);
        if (candidate == NULL) {
            continue;
        }

        if (strcmp(candidate->name, property_name) == 0) {
            if (property_id != NULL) {
                *property_id = candidate->prop_id;
            }
            if (value != NULL) {
                *value = object_properties->prop_values[i];
            }
            if (property != NULL) {
                *property = candidate;
            } else {
                drmModeFreeProperty(candidate);
            }
            found = true;
            break;
        }

        drmModeFreeProperty(candidate);
    }

    drmModeFreeObjectProperties(object_properties);
    return found;
}

// 轮询并处理 DRM 文件描述符上的事件（主要用于消费 VSYNC 带来的 Page Flip 回调）
bool DrmDisplay::PumpEvents(int timeout_ms) {
    drmEventContext event_context;
    memset(&event_context, 0, sizeof(event_context));
    event_context.version = DRM_EVENT_CONTEXT_VERSION;
    event_context.page_flip_handler = &DrmDisplay::PageFlipHandler;

    struct pollfd poll_fd;
    memset(&poll_fd, 0, sizeof(poll_fd));
    poll_fd.fd = drm_fd_;
    poll_fd.events = POLLIN;

    int ret = poll(&poll_fd, 1, timeout_ms);
    if (ret < 0) {
        printf("drm_display: poll failed\n");
        return false;
    }
    if (ret == 0) {
        return true;
    }
    if (poll_fd.revents & POLLIN) {
        if (drmHandleEvent(drm_fd_, &event_context) != 0) {
            printf("drm_display: drmHandleEvent failed\n");
            return false;
        }
    }

    return true;
}

// 基于 Overlay 模式执行真正的渲染逻辑：格式转换、画框、提交 Atomic 请求来翻页
bool DrmDisplay::PresentWithOverlay(MppFrameHandle frame,
                                    const object_detect_result_list &detections,
                                    double render_fps,
                                    double latency_ms) {
    if (!PumpEvents(2)) {
        return false;
    }
    if (page_flip_pending_) {
        return true;
    }

    letterbox_t letter_box;
    memset(&letter_box, 0, sizeof(letter_box));
    if (!ComputeLetterboxLayout(frame.image,
                                mode_.hdisplay,
                                mode_.vdisplay,
                                &letter_box)) {
        return false;
    }

    int buffer_index = overlay_started_ ? (front_buffer_index_ + 1) % kBufferCount : 0;
    DisplayBuffer &buffer = buffers_[buffer_index];
    if (buffer.map == NULL) {
        return false;
    }

    int ret = convert_image_with_letterbox(const_cast<image_buffer_t *>(&frame.image),
                                           &buffer.rga_image,
                                           &letter_box,
                                           0);
    if (ret != 0) {
        printf("drm_display: convert_image_with_letterbox failed for overlay plane, ret=%d\n", ret);
        return false;
    }
    if (dma_sync_device_to_cpu(buffer.prime_fd) != 0) {
        printf("drm_display: dma_sync_device_to_cpu failed for overlay buffer\n");
        return false;
    }
    DrawDetections(&buffer.cpu_image, detections, letter_box, render_fps, latency_ms);
    if (dma_sync_cpu_to_device(buffer.prime_fd) != 0) {
        printf("drm_display: dma_sync_cpu_to_device failed for overlay buffer\n");
        return false;
    }

    if (!overlay_started_) {
        if (!AtomicCommitOverlay(buffer_index, 0, NULL)) {
            printf("drm_display: initial overlay plane commit failed errno=%d(%s)\n",
                   errno,
                   strerror(errno));
            return false;
        }

        front_buffer_index_ = buffer_index;
        overlay_started_ = true;
        return true;
    }

    if (!AtomicCommitOverlay(buffer_index,
                             DRM_MODE_PAGE_FLIP_EVENT | DRM_MODE_ATOMIC_NONBLOCK,
                             this)) {
        if (errno == EBUSY || errno == EINTR) {
            return true;
        }
        printf("drm_display: atomic overlay commit failed errno=%d(%s)\n",
               errno,
               strerror(errno));
        return false;
    }

    pending_buffer_index_ = buffer_index;
    page_flip_pending_ = true;
    return true;
}

// 构造 DRM Atomic 提交请求，将选定的 Dumb Buffer 绑定到 Overlay Plane 上显示
bool DrmDisplay::AtomicCommitOverlay(int buffer_index,
                                     uint32_t flags,
                                     void *user_data) {
    drmModeAtomicReq *request = drmModeAtomicAlloc();
    if (request == NULL) {
        errno = ENOMEM;
        return false;
    }

    bool success = true;
    success = success &&
              AddPlaneToAtomicRequest(request,
                                      video_plane_,
                                      buffers_[buffer_index].fb_id,
                                      0,
                                      0,
                                      mode_.hdisplay,
                                      mode_.vdisplay,
                                      mode_.hdisplay,
                                      mode_.vdisplay);

    int ret = success
                  ? drmModeAtomicCommit(drm_fd_, request, flags, user_data)
                  : -1;
    drmModeAtomicFree(request);
    return ret == 0;
}

// 将保存的 Plane 初始状态参数加载进 Atomic 请求中，用于程序退出时恢复桌面原貌
bool DrmDisplay::AddRestorePlaneToAtomicRequest(drmModeAtomicReqPtr request,
                                                const PlaneState &plane_state) const {
    if (request == NULL || !plane_state.valid()) {
        return false;
    }

    if (drmModeAtomicAddProperty(request,
                                 plane_state.plane_id,
                                 plane_state.props.crtc_id,
                                 plane_state.current_crtc_id) < 0 ||
        drmModeAtomicAddProperty(request,
                                 plane_state.plane_id,
                                 plane_state.props.fb_id,
                                 plane_state.current_fb_id) < 0 ||
        drmModeAtomicAddProperty(request,
                                 plane_state.plane_id,
                                 plane_state.props.crtc_x,
                                 plane_state.current_crtc_x) < 0 ||
        drmModeAtomicAddProperty(request,
                                 plane_state.plane_id,
                                 plane_state.props.crtc_y,
                                 plane_state.current_crtc_y) < 0 ||
        drmModeAtomicAddProperty(request,
                                 plane_state.plane_id,
                                 plane_state.props.crtc_w,
                                 plane_state.current_crtc_w) < 0 ||
        drmModeAtomicAddProperty(request,
                                 plane_state.plane_id,
                                 plane_state.props.crtc_h,
                                 plane_state.current_crtc_h) < 0 ||
        drmModeAtomicAddProperty(request,
                                 plane_state.plane_id,
                                 plane_state.props.src_x,
                                 plane_state.current_src_x) < 0 ||
        drmModeAtomicAddProperty(request,
                                 plane_state.plane_id,
                                 plane_state.props.src_y,
                                 plane_state.current_src_y) < 0 ||
        drmModeAtomicAddProperty(request,
                                 plane_state.plane_id,
                                 plane_state.props.src_w,
                                 plane_state.current_src_w) < 0 ||
        drmModeAtomicAddProperty(request,
                                 plane_state.plane_id,
                                 plane_state.props.src_h,
                                 plane_state.current_src_h) < 0) {
        return false;
    }

    if (plane_state.has_alpha &&
        drmModeAtomicAddProperty(request,
                                 plane_state.plane_id,
                                 plane_state.props.alpha,
                                 plane_state.current_alpha) < 0) {
        return false;
    }
    if (plane_state.has_pixel_blend_mode &&
        drmModeAtomicAddProperty(request,
                                 plane_state.plane_id,
                                 plane_state.props.pixel_blend_mode,
                                 plane_state.current_pixel_blend_mode) < 0) {
        return false;
    }

    return true;
}

// 将渲染需要的新 FB 参数及显示坐标（源区域与目标区域）组装进 Atomic 请求中
bool DrmDisplay::AddPlaneToAtomicRequest(drmModeAtomicReqPtr request,
                                         const PlaneState &plane_state,
                                         uint32_t fb_id,
                                         int crtc_x,
                                         int crtc_y,
                                         int crtc_w,
                                         int crtc_h,
                                         int src_w,
                                         int src_h) const {
    if (request == NULL || !plane_state.valid()) {
        return false;
    }

    if (drmModeAtomicAddProperty(request,
                                 plane_state.plane_id,
                                 plane_state.props.crtc_id,
                                 crtc_id_) < 0 ||
        drmModeAtomicAddProperty(request,
                                 plane_state.plane_id,
                                 plane_state.props.fb_id,
                                 fb_id) < 0 ||
        drmModeAtomicAddProperty(request,
                                 plane_state.plane_id,
                                 plane_state.props.crtc_x,
                                 static_cast<uint64_t>(crtc_x)) < 0 ||
        drmModeAtomicAddProperty(request,
                                 plane_state.plane_id,
                                 plane_state.props.crtc_y,
                                 static_cast<uint64_t>(crtc_y)) < 0 ||
        drmModeAtomicAddProperty(request,
                                 plane_state.plane_id,
                                 plane_state.props.crtc_w,
                                 static_cast<uint64_t>(crtc_w)) < 0 ||
        drmModeAtomicAddProperty(request,
                                 plane_state.plane_id,
                                 plane_state.props.crtc_h,
                                 static_cast<uint64_t>(crtc_h)) < 0 ||
        drmModeAtomicAddProperty(request,
                                 plane_state.plane_id,
                                 plane_state.props.src_x,
                                 0) < 0 ||
        drmModeAtomicAddProperty(request,
                                 plane_state.plane_id,
                                 plane_state.props.src_y,
                                 0) < 0 ||
        drmModeAtomicAddProperty(request,
                                 plane_state.plane_id,
                                 plane_state.props.src_w,
                                 static_cast<uint64_t>(src_w) << 16) < 0 ||
        drmModeAtomicAddProperty(request,
                                 plane_state.plane_id,
                                 plane_state.props.src_h,
                                 static_cast<uint64_t>(src_h) << 16) < 0) {
        return false;
    }

    if (plane_state.has_alpha &&
        drmModeAtomicAddProperty(request,
                                 plane_state.plane_id,
                                 plane_state.props.alpha,
                                 plane_state.alpha_opaque) < 0) {
        return false;
    }
    if (plane_state.has_pixel_blend_mode &&
        drmModeAtomicAddProperty(request,
                                 plane_state.plane_id,
                                 plane_state.props.pixel_blend_mode,
                                 plane_state.blend_mode_value) < 0) {
        return false;
    }

    return true;
}

// 在 CPU 映射的图像上绘制对象检测结果（绿色边界框和红色分类标签/概率），并绘制当前运行的帧率及延迟状态
void DrmDisplay::DrawDetections(image_buffer_t *image,
                                const object_detect_result_list &detections,
                                const letterbox_t &letter_box,
                                double render_fps,
                                double latency_ms) const {
    if (image == NULL || image->virt_addr == NULL) {
        return;
    }

    for (int i = 0; i < detections.count; ++i) {
        const object_detect_result &det = detections.results[i];
        int left = static_cast<int>(det.box.left * letter_box.scale + letter_box.x_pad);
        int top = static_cast<int>(det.box.top * letter_box.scale + letter_box.y_pad);
        int right = static_cast<int>(det.box.right * letter_box.scale + letter_box.x_pad);
        int bottom = static_cast<int>(det.box.bottom * letter_box.scale + letter_box.y_pad);

        draw_rectangle(image,
                       left,
                       top,
                       right - left,
                       bottom - top,
                       COLOR_GREEN,
                       2);

        char text[128];
        snprintf(text,
                 sizeof(text),
                 "%s %.1f%%",
                 coco_cls_to_name(det.cls_id),
                 det.prop * 100.0f);
        DrawTextLine(image,
                     text,
                     left,
                     top > 24 ? top - 20 : top + 6,
                     COLOR_RED,
                     16);
    }

    char stats_text[128];
    snprintf(stats_text,
             sizeof(stats_text),
             "FPS %.2f  Latency %.2f ms",
             render_fps,
             latency_ms);
    DrawTextLine(image, stats_text, 12, 12, COLOR_YELLOW, 20);
}

// DRM Page Flip 事件的回调处理函数，当显卡完成缓存交换(VSYNC)后触发，用于更新前台和挂起缓冲区的索引
void DrmDisplay::PageFlipHandler(int fd,
                                 unsigned int frame,
                                 unsigned int sec,
                                 unsigned int usec,
                                 void *data) {
    (void)fd;
    (void)frame;
    (void)sec;
    (void)usec;

    DrmDisplay *display = static_cast<DrmDisplay *>(data);
    if (display == NULL) {
        return;
    }

    if (display->pending_buffer_index_ >= 0 &&
        display->pending_buffer_index_ < kBufferCount) {
        display->front_buffer_index_ = display->pending_buffer_index_;
    }
    display->pending_buffer_index_ = -1;
    display->page_flip_pending_ = false;
}

}  // namespace rknn_demo