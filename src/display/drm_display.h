#ifndef RKNN_YOLOV5_DEMO_DISPLAY_DRM_DISPLAY_H_
#define RKNN_YOLOV5_DEMO_DISPLAY_DRM_DISPLAY_H_

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <string>
#include <vector>

#include <xf86drmMode.h>

#include "image_utils.h"
#include "model/postprocess.h"
#include "pipeline/pipeline_types.h"

// 文件说明：
// 封装基于 DRM/KMS 的本地显示输出，只申请 Overlay Plane 做叠加显示，
// 不占用系统 Primary Plane，也不主动接管整块屏幕。

namespace rknn_demo {

class DrmDisplay {
public:
    // 构造函数：初始化所有成员变量的默认值
    DrmDisplay();
    // 析构函数：确保在对象销毁时释放资源并恢复 DRM 状态
    ~DrmDisplay();

    // 初始化 DRM 显示子系统（打开设备、分配缓冲、探测显示器和 Plane）
    bool Init();
    
    // 提交一帧图像及目标检测结果到显示设备
    bool Present(MppFrameHandle frame,
                 const object_detect_result_list &detections,
                 double render_fps,
                 double latency_ms);
                 
    // 关闭显示设备，释放内部缓冲，并还原系统原本的桌面/显示状态
    void Close();

private:
    // 双缓冲机制的缓冲区数量（2 表示 Ping-Pong Buffer）
    static const int kBufferCount = 2;

    // 内部结构体：封装用于显示的缓冲区及其相关的句柄和映射地址
    struct DisplayBuffer {
        uint32_t handle;      // DRM 内部管理的 GEM 内存句柄
        uint32_t fb_id;       // 注册到 DRM 的 Framebuffer ID
        uint32_t pitch;       // 图像的行跨度（每行的字节数）
        int prime_fd;         // 导出的 dma-buf 文件描述符，供 RGA 或其他硬件模块共享
        void *map;            // 映射到用户空间的 CPU 虚拟地址
        size_t size;          // 缓冲区分配的总字节大小
        image_buffer_t rga_image; // 封装好的图像结构，专供 RGA 硬件加速模块使用
        image_buffer_t cpu_image; // 封装好的图像结构，专供 CPU 绘图（如画框、写字）使用

        DisplayBuffer()
            : handle(0),
              fb_id(0),
              pitch(0),
              prime_fd(-1),
              map(NULL),
              size(0) {
            memset(&rga_image, 0, sizeof(rga_image));
            memset(&cpu_image, 0, sizeof(cpu_image));
            rga_image.fd = -1;
            cpu_image.fd = -1;
        }
    };

    // 内部结构体：保存 Connector（连接器）的关键属性 ID
    struct ConnectorProperties {
        uint32_t crtc_id; // Connector 绑定的 CRTC ID 属性

        ConnectorProperties() : crtc_id(0) {}
    };

    // 内部结构体：保存 CRTC（显示控制器）的关键属性 ID
    struct CrtcProperties {
        uint32_t active;  // CRTC 是否激活的属性 ID
        uint32_t mode_id; // CRTC 当前显示模式 (Mode blob) 的属性 ID

        CrtcProperties() : active(0), mode_id(0) {}
    };

    // 内部结构体：保存 Plane（图层）的所有关键属性 ID
    struct PlaneProperties {
        uint32_t type;             // 图层类型 (Primary, Overlay, Cursor) 的属性 ID
        uint32_t crtc_id;          // 图层绑定的 CRTC ID 属性 ID
        uint32_t fb_id;            // 图层绑定的 Framebuffer ID 属性 ID
        uint32_t crtc_x;           // 图层在 CRTC(屏幕) 上显示的起始 X 坐标属性 ID
        uint32_t crtc_y;           // 图层在 CRTC(屏幕) 上显示的起始 Y 坐标属性 ID
        uint32_t crtc_w;           // 图层在 CRTC(屏幕) 上显示的宽度属性 ID
        uint32_t crtc_h;           // 图层在 CRTC(屏幕) 上显示的高度属性 ID
        uint32_t src_x;            // 裁剪源图像的起始 X 坐标 (16.16定点数) 属性 ID
        uint32_t src_y;            // 裁剪源图像的起始 Y 坐标 (16.16定点数) 属性 ID
        uint32_t src_w;            // 裁剪源图像的宽度 (16.16定点数) 属性 ID
        uint32_t src_h;            // 裁剪源图像的高度 (16.16定点数) 属性 ID
        uint32_t alpha;            // 图层全局透明度属性 ID
        uint32_t pixel_blend_mode; // 图层像素混合模式 (如 Pre-multiplied, Coverage) 属性 ID

        PlaneProperties()
            : type(0),
              crtc_id(0),
              fb_id(0),
              crtc_x(0),
              crtc_y(0),
              crtc_w(0),
              crtc_h(0),
              src_x(0),
              src_y(0),
              src_w(0),
              src_h(0),
              alpha(0),
              pixel_blend_mode(0) {}
    };

    // 内部结构体：用于缓存一个 Plane 当前的所有属性 ID 及其对应的属性值
    struct PlaneState {
        uint32_t plane_id;                  // 图层的物理 ID
        uint64_t type_value;                // 图层具体类型值
        uint64_t alpha_opaque;              // 完全不透明时的 Alpha 值配置
        uint64_t blend_mode_value;          // 图层混合模式的具体配置值
        uint64_t current_crtc_id;           // 当前绑定的 CRTC ID 值
        uint64_t current_fb_id;             // 当前绑定的 Framebuffer ID 值
        uint64_t current_crtc_x;            // 当前在屏幕上的 X 坐标值
        uint64_t current_crtc_y;            // 当前在屏幕上的 Y 坐标值
        uint64_t current_crtc_w;            // 当前在屏幕上的宽度值
        uint64_t current_crtc_h;            // 当前在屏幕上的高度值
        uint64_t current_src_x;             // 当前源图像裁剪 X 坐标值
        uint64_t current_src_y;             // 当前源图像裁剪 Y 坐标值
        uint64_t current_src_w;             // 当前源图像裁剪宽度值
        uint64_t current_src_h;             // 当前源图像裁剪高度值
        uint64_t current_alpha;             // 当前透明度值
        uint64_t current_pixel_blend_mode;  // 当前混合模式值
        bool has_alpha;                     // 标志该图层是否支持 alpha 透明度调节
        bool has_pixel_blend_mode;          // 标志该图层是否支持像素级别的混合模式调节
        PlaneProperties props;              // 缓存的各类属性的标识符 (Property IDs)

        PlaneState()
            : plane_id(0),
              type_value(0),
              alpha_opaque(0),
              blend_mode_value(0),
              current_crtc_id(0),
              current_fb_id(0),
              current_crtc_x(0),
              current_crtc_y(0),
              current_crtc_w(0),
              current_crtc_h(0),
              current_src_x(0),
              current_src_y(0),
              current_src_w(0),
              current_src_h(0),
              current_alpha(0),
              current_pixel_blend_mode(0),
              has_alpha(false),
              has_pixel_blend_mode(false) {}

        // 判断该 Plane 状态结构体是否合法有效
        bool valid() const {
            return plane_id != 0;
        }
    };

    // 遍历系统的显卡节点并寻找可用的 KMS 设备
    bool OpenDevice();
    // 尝试打开并验证特定路径下的 DRM 节点
    bool OpenDeviceAtPath(const char *path);
    // 从已打开的 DRM 设备中自动选择连通的连接器 (Connector) 和匹配的 CRTC
    bool SelectConnectorAndCrtc();
    // 创建用于 DRM 扫描输出的 Dumb Buffers
    bool CreateScanoutBuffers();
    // 捕获当前的 DRM 桌面状态，用于程序退出时恢复原貌
    bool CaptureRestoreState();
    // 找到并初始化一个可以用于视频显示的 Overlay Plane
    bool InitOverlayPlanes();
    // 清空与重置 Overlay Plane 的缓存状态变量
    void ResetOverlayPlaneState();
    // 等待挂起的 Page Flip (翻页) 事件完成，避免画面撕裂
    bool WaitForPendingFlip(int timeout_ms);
    // 执行显示状态还原操作 (还原原有的 Plane、CRTC 配置)
    bool RestoreDisplayState();
    // 根据需求的图层类型和像素格式，筛选出一个合适的 Plane
    bool SelectPlaneForFormat(uint64_t desired_type,
                              uint32_t fourcc,
                              uint32_t excluded_plane_id,
                              PlaneState *plane_state);
    // 获取并解析指定 Plane 的所有 Atomic 属性状态
    bool LoadPlaneState(uint32_t plane_id, PlaneState *plane_state) const;
    // 通用函数：查询 DRM 对象的具体属性（获取 property ID 和 value）
    bool GetObjectProperty(uint32_t object_id,
                           uint32_t object_type,
                           const char *property_name,
                           uint32_t *property_id,
                           uint64_t *value,
                           drmModePropertyRes **property) const;
    // 轮询 DRM 事件循环，触发和消费 VSYNC/Page Flip 等回调事件
    bool PumpEvents(int timeout_ms);
    // 使用 Overlay Plane 来渲染带有检测框的图像并提交上屏
    bool PresentWithOverlay(MppFrameHandle frame,
                            const object_detect_result_list &detections,
                            double render_fps,
                            double latency_ms);
    // 将指定缓冲区的状态通过 DRM Atomic 接口进行提交
    bool AtomicCommitOverlay(int buffer_index,
                             uint32_t flags,
                             void *user_data);
    // 将捕获的图层原有参数添加到 Atomic 恢复请求中
    bool AddRestorePlaneToAtomicRequest(drmModeAtomicReqPtr request,
                                        const PlaneState &plane_state) const;
    // 将新图层的目标和源区域参数添加到 Atomic 渲染请求中
    bool AddPlaneToAtomicRequest(drmModeAtomicReqPtr request,
                                 const PlaneState &plane_state,
                                 uint32_t fb_id,
                                 int crtc_x,
                                 int crtc_y,
                                 int crtc_w,
                                 int crtc_h,
                                 int src_w,
                                 int src_h) const;
    // 在 CPU 映射的显存上绘制检测框、类别名称、以及性能指标文本
    void DrawDetections(image_buffer_t *image,
                        const object_detect_result_list &detections,
                        const letterbox_t &letter_box,
                        double render_fps,
                        double latency_ms) const;

    // DRM 底层抛出的 Page Flip 完成事件的静态回调函数
    static void PageFlipHandler(int fd,
                                unsigned int frame,
                                unsigned int sec,
                                unsigned int usec,
                                void *data);

    int drm_fd_;                                     // DRM 节点设备的文件描述符
    std::string drm_card_path_;                      // 成功打开的 DRM 设备路径 (如 /dev/dri/card0)
    uint32_t connector_id_;                          // 使用的连接器 ID
    uint32_t crtc_id_;                               // 使用的 CRTC ID
    int crtc_index_;                                 // CRTC 在 DRM 资源列表中的数组索引
    drmModeModeInfo mode_;                           // 当前输出屏幕的显示模式信息 (分辨率、刷新率)
    drmModeCrtc *saved_crtc_;                        // 启动前保存的原有 CRTC 状态结构体
    DisplayBuffer buffers_[kBufferCount];            // 循环使用的双/多缓冲区队列
    int front_buffer_index_;                         // 当前正在前台（屏幕上）显示的缓冲区索引
    int pending_buffer_index_;                       // 已提交给内核等待下一次 VSYNC 翻转的缓冲区索引
    bool page_flip_pending_;                         // 是否存在正在等待硬件翻转 (Page Flip) 的标志位
    bool initialized_;                               // 整个显示模块是否成功初始化的标志位
    bool overlay_started_;                           // Overlay Plane 是否已经进行了首次提交上屏的标志位
    bool restore_state_captured_;                    // 是否已成功抓取了程序的初始桌面状态
    uint64_t saved_connector_crtc_id_value_;         // 启动前捕获的 Connector 关联的原始 CRTC ID 值
    ConnectorProperties connector_props_;            // 缓存的 Connector 相关属性 ID
    CrtcProperties crtc_props_;                      // 缓存的 CRTC 相关属性 ID
    PlaneState video_plane_;                         // 用于此次应用视频输出叠加层的图层状态
    std::vector<PlaneState> saved_plane_states_;     // 捕获的全部图层初始状态记录，退出时需全部还原
};

}  // namespace rknn_demo

#endif  // RKNN_YOLOV5_DEMO_DISPLAY_DRM_DISPLAY_H_