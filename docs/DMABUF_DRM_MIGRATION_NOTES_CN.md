# RK3588 DMA-BUF / DRM 改造踩坑记录

## 背景

这份文档记录本项目从

`FFmpeg demux + FFmpeg rkmpp decode + CPU/AVFrame 传递 + OpenCV 显示`

迁移到

`FFmpeg demux + 纯 MPP decode(dma-buf) + RGA(fd->fd) + RKNN io_mem(fd) + DRM/KMS 显示`

过程中遇到的主要问题、根因、修复方式和经验总结。

目标读者：

- 后续继续维护这条链路的人
- 想把其他项目迁移到 `DMA-BUF` 方案的人
- 需要快速判断“卡在哪一层”的排障人员

---

## 当前链路

当前已经跑通的主链路是：

1. `FFmpeg demux`
   负责 `RTSP` 会话、网络收包和 `AVPacket` 输出。

2. `纯 MPP decode`
   接收压缩码流，输出 `MppFrame + dma-buf fd`。

3. `RGA`
   直接读取解码 `fd`，写入推理输入 `fd`。

4. `RKNN io_mem`
   使用 `rknn_create_mem_from_fd + rknn_set_io_mem` 把 DMA-BUF 绑定为模型输入。

5. `CPU 后处理`
   当前检测框后处理仍然在 CPU 上完成。

6. `DRM/KMS 显示`
   只申请一个 `Overlay Plane` 做叠加显示；
   不占用系统 `Primary Plane`，退出时优先恢复进入程序前的 plane 状态。

---

## 踩坑列表

### 1. 程序只打印模型信息，然后看起来“卡死”

现象：

- 模型输入输出属性打印完成后没有后续画面
- 终端也没有新的统计输出

根因：

- 原始工程在 `capture / decode / infer / render` 四段缺少首帧定位日志
- 一旦卡在某一级，外部看起来都像“程序不动了”

修复：

- 为 `demux / decode / infer / render` 增加一次性阶段日志
- 加上 RTSP 打开、首帧解码、首个推理结果等关键提示

经验：

- 多线程流水线项目，一开始就要有“首包 / 首帧 / 首结果”日志
- 否则定位速度会非常慢

---

### 2. 显示失败时，程序表面上像完全不工作

现象：

- 没画面
- 也没统计输出
- 但进程并没有退出

根因：

- 旧 `render` 路径中，显示帧转换失败后直接 `continue`
- 结果统计线程仍在跑，但没有任何可见输出

修复：

- 显示初始化失败时自动降级为 headless stats 模式
- 显示转换失败时只关闭显示，不影响统计打印

经验：

- 显示失败不能把“统计线程”一并静默掉
- 可视化是附加能力，统计和主链路不能依赖它

---

### 3. RTSP 打开成功，但收包失败时没有足够信息

现象：

- `RTSP` 地址看起来能打开
- 但后续没有帧，定位困难

根因：

- `av_read_frame` 失败时没有明确日志
- 网络超时参数也不完整

修复：

- 增加 `rw_timeout`
- 对 `av_read_frame` 的失败做显式日志输出

经验：

- 网络流必须带 timeout
- `open success` 不代表 `read success`

---

### 4. `ffmpeg + rkmpp` 输出的 `NV12` 不是紧凑连续布局

现象：

- 能解码出首帧
- 但推理线程一直进不去
- 最后定位到 `BuildImageBufferFromAvFrame()` 失败

根因：

- 旧代码假设：
  `data[1] == data[0] + linesize[0] * height`
- 实际 `ffmpeg + rkmpp` 输出的 `AVFrame` 往往带对齐 padding
- `UV` 平面不一定紧贴 `Y` 平面

修复：

- 先放宽 `AVFrame -> image_buffer_t` 的布局判断
- 再进一步把非紧凑帧重新打包成连续内存

经验：

- `NV12` 不等于“线性紧凑内存”
- 只看像素格式不够，必须同时看 stride / plane delta

---

### 5. `av_frame_get_buffer()` 不能保证得到“推理想要的紧凑 NV12”

现象：

- 已经尝试把解码帧重新打包
- 但打包后仍可能出现不符合下游假设的布局

根因：

- `av_frame_get_buffer()` 自己也会按对齐策略分配平面
- 它保证的是 FFmpeg 语义，不是“Y+UV 完全紧凑拼接”

修复：

- 改为显式：
  - `av_image_get_buffer_size(..., align=1)`
  - `av_malloc`
  - `av_image_fill_arrays(..., align=1)`
  - `av_image_copy`

经验：

- 想要“严格连续平面内存”时，不要依赖默认 buffer allocator

---

### 6. 想临时绕开 RGA，结果 CPU fallback 也跑不通

现象：

- 加了“禁用 RGA”的调试逻辑后，推理直接失败
- 报 `convert_image_with_letterbox fail`

根因：

- 当前 CPU fallback 主要支持：
  - 同格式
  - `RGB <-> BGR`
- 不支持 `NV12 -> RGB888`

修复：

- 删除“自动绕开 RGA”的默认策略
- 仅保留显式环境变量作为调试开关

经验：

- 排查时临时绕开的路径，必须先确认功能等价
- 否则会引入第二个假问题

---

### 7. 纯 MPP 解码时，误把压缩码流输入也改成了 MPP buffer

现象：

- 板端出现：
  - `allocator_std_open Warning`
  - `mpp_buffer_create failed`

根因：

- 把输入侧也误改成了 `mpp_buffer_get + mpp_packet_init_with_buffer`
- 对普通视频压缩包输入来说，这不是正确姿势

修复：

- 改回标准方式：
  - `AVPacket.data -> mpp_packet_init(...)`
- 只让“解码输出帧”走 DMA-BUF

经验：

- `DMA-BUF` 化的重点是“解码后 raw frame”
- 压缩码流输入不需要强行 DMA-BUF 化

---

### 8. 纯 MPP 解码真正关键的是 `INFO_CHANGE + EXT_BUF_GROUP`

现象：

- MPP 初始化成功
- 但不一定能稳定输出可复用的 DRM buffer

根因：

- MPP 在拿到首个 `info_change` 帧前，真正输出尺寸和 stride 可能还没确定
- 不处理 `MPP_DEC_SET_EXT_BUF_GROUP` / `MPP_DEC_SET_INFO_CHANGE_READY`，输出路径会不稳定

修复：

- 在 `info_change` 时：
  - 建立 `DRM` 类型输出 buffer group
  - `MPP_DEC_SET_EXT_BUF_GROUP`
  - `MPP_DEC_SET_INFO_CHANGE_READY`

经验：

- 纯 MPP 解码不是 `mpp_init()` 完就完事
- `info_change` 是必须认真处理的阶段

---

### 9. `RKNN` 输入不能再走 `malloc + rknn_inputs_set`

现象：

- 解码到 RGA 已经全是 `fd`
- 但进入 RKNN 时如果还走 CPU buffer，就把 zero-copy 断掉了

根因：

- 旧模型代码使用：
  - `malloc model_input_buffer`
  - `rknn_inputs_set(buf=virt_addr)`

修复：

- 改为：
  - `dma_buf_alloc`
  - `rknn_create_mem_from_fd`
  - `rknn_set_io_mem`
- 并结合 `w_stride / size_with_stride / h_stride`

经验：

- 真正的 `DMA-BUF -> NPU`，核心不是 `rknn_run`
- 而是 `rknn_set_io_mem`

---

### 10. 不能忽略 `size_with_stride / w_stride / h_stride`

现象：

- 申请的输入缓冲可能够“像素数”，但仍然不稳

根因：

- RKNN 输入 tensor 有自己的 stride 语义
- 只按 `width * height * channel` 申请不一定足够

修复：

- 输入 DMA-BUF 尺寸优先用：
  - `input_attr.size_with_stride`
- `width_stride` 优先用：
  - `input_attr.w_stride`
- `h_stride` 显式回填到 `rknn_tensor_attr`

经验：

- NPU tensor 的“真实内存尺寸”不等于逻辑尺寸

---

### 11. 旧 `render` 仍然依赖 `AVFrame/OpenCV`

现象：

- 解码和推理已经改成 `MppFrame + dma-buf`
- 但显示仍然走 `OpenCV imshow`
- 一开显示就不兼容

根因：

- 原始 `render` 路径依赖 `AvFrameHandle -> cv::Mat`
- 新链路下渲染侧拿到的是 `MppFrameHandle`

修复：

- 新增 DRM/KMS 显示器
- 不再依赖 `OpenCV` 弹窗显示

经验：

- 只改“解码到推理”还不够
- 只要显示还在 `Mat/CPU` 语义里，链路就不完整

---

### 12. 单缓冲 DRM 显示会严重拉高 `wait_render`

现象：

- 屏幕能显示
- 但 `wait_render` 高
- `drop_rnd` 很快增加

根因：

- 单个 dumb buffer 同时承担：
  - 扫描输出
  - RGA 写入
  - CPU 画框
- 渲染线程会和显示刷新互相争用

修复：

- 升级到双缓冲

经验：

- DRM 显示要做实时视频，单缓冲几乎一定不够

---

### 13. 双缓冲如果“同步等待 page-flip”，性能仍然会崩

现象：

- 逻辑上更正确了
- 但 FPS 反而掉到 `12~14`
- `wait_render` 到 `90ms+`

根因：

- 每一帧都：
  - `QueuePageFlip`
  - `WaitPageFlip`
- 渲染线程被 vsync 节奏硬阻塞

修复：

- 改成异步 page-flip：
  - 非阻塞处理一次 DRM event
  - 如果上一帧 flip 未完成，直接跳过当前显示提交

经验：

- 低延迟显示链的原则不是“每帧都显示”
- 而是“显示永远不能反压推理”

---

### 14. DRM/KMS 路径需要接受“显示跳帧”

现象：

- 异步 page-flip 后，显示不一定每帧都上屏

根因：

- 为了保证主链路吞吐，显示帧要允许丢弃旧帧

修复：

- 使用“最新帧优先”策略
- page-flip 忙时直接跳过当前显示帧

经验：

- `drop_rnd` 不一定是 bug
- 对实时系统来说，它经常是“主动低延迟策略”的体现

---

### 15. 多推理线程会重复打印模型 tensor 属性

现象：

- `--infer-threads 3` 时，输入输出 tensor 属性会打印三遍

根因：

- 每个 worker 都独立初始化一套 RKNN context

修复建议：

- 后续可只保留 `worker 0` 打印

经验：

- 这不是功能错误
- 但会显著增加日志噪音

---

### 16. 显示统计 FPS 看起来正常，但肉眼看起来非常卡

现象：

- 终端统计 `fps` 仍然接近 25
- 但屏幕上肉眼看到的画面明显不流畅
- 主观感觉比日志反映的要卡很多

根因：

- 统计口径是“渲染线程消费结果包的速度”
- 不是“面板真正显示新帧的速度”
- 显示链初版中：
  - RGA 写 dumb buffer
  - CPU 再画框
  - KMS 扫描显示
- 这三者之间如果同步策略不对，会出现：
  - 后台统计快
  - 前台显示不顺

修复：

- 改为双缓冲
- 再进一步改成异步 page-flip

经验：

- “rendered FPS” 不一定等于“用户肉眼看到的刷新体验”
- 显示链路必须单独优化

---

### 17. 只做双缓冲还不够，同步等待 page-flip 会把显示拖成瓶颈

现象：

- 双缓冲后，功能正常
- 但 FPS 一度下降到 `12~14`
- `wait_render` 飙升到 `90ms+`

根因：

- 双缓冲版本里，每一帧都同步等待 `page flip` 完成
- render 线程被显示刷新节奏硬阻塞

修复：

- 把显示改成异步 page-flip
- 如果上一帧 flip 还没结束，直接跳过当前显示更新

经验：

- 双缓冲不是万能药
- 如果使用方式是“阻塞等 vsync”，一样会严重拖慢主链

---

### 18. DRM dumb buffer 上先 RGA 后 CPU 画框，必须做缓存同步

现象：

- 统计正常
- 但屏幕上画面卡顿、撕裂感强，或者过一段时间突然消失
- 后台推理和统计依然继续

根因：

- dumb buffer 同时被：
  - RGA 作为设备写端访问
  - CPU 作为映射内存访问
  - KMS 作为扫描端访问
- 如果没有 `device -> cpu` / `cpu -> device` 同步，
  三方看到的内容可能不一致

修复：

- RGA 写完后：
  - `dma_sync_device_to_cpu`
- CPU 画框后：
  - `dma_sync_cpu_to_device`

经验：

- `DMA-BUF` 不等于天然缓存一致
- 设备和 CPU 混合访问时，显式同步通常是必须的

---

### 19. DRM event pump 不能完全零超时轮询

现象：

- 偶发 page-flip 状态处理不及时
- 显示节奏不稳

根因：

- 完全 `0ms poll` 的 event pump 在某些板端上太激进
- 容易把 page-flip 事件处理窗口压得过紧

修复：

- 改成一个很短的超时轮询

经验：

- 显示事件循环不应完全忙轮询
- 极短超时通常比 0ms 更稳

---

### 20. `drmModePageFlip` 偶发失败时，不能直接永久关闭显示

现象：

- 程序长时间运行后偶发：
  - `drmModePageFlip failed`
- 旧逻辑一旦失败，就把显示永久降级成 headless
- 结果是：
  - 后台仍在推理和打印 stats
  - 画面彻底黑掉

根因：

- page-flip 失败并不总是“显示链彻底坏了”
- 有些只是瞬时忙、被打断，或者一次临时状态异常

修复：

- `EBUSY / EINTR`
  - 直接当作“当前显示帧丢弃”
  - 不关闭显示
- `EINVAL`
  - 先尝试退回一次 `drmModeSetCrtc` 恢复
  - 只有恢复失败才真正关闭显示

经验：

- DRM page-flip 的错误需要分级处理
- 不是所有失败都值得“一票否决”

---

### 21. 最终稳定策略是“显示允许丢帧，但不能拖慢推理”

现象：

- 异步 page-flip 后，显示仍可能偶尔跳帧
- 但整体吞吐和延迟显著改善

根因：

- 实时检测系统里，显示是附属链路
- 目标应当是“保证主链路低延迟”，而不是“保证每帧都上屏”

修复：

- 使用“最新帧优先”
- page-flip 忙时直接跳过当前显示帧
- 不再阻塞推理线程或 render 线程等待显示

经验：

- 对实时视频检测来说：
  - 允许显示跳帧是合理的
  - 禁止显示反压主链才是关键

---

### 22. `Ctrl+C` 退出时必须走“可恢复”的 DRM 清理路径

现象：

- 程序被 `Ctrl+C` 打断后，偶发：
  - 屏幕保持黑屏
  - plane / CRTC 状态没有恢复
  - DRM Master 没及时释放
  - 下次启动显示初始化失败

根因：

- 旧版本主要依赖进程直接退出
- 如果线程仍卡在：
  - `avformat_open_input`
  - `av_read_frame`
  - 队列阻塞等待
- 就可能来不及走到 `DrmDisplay::Close()` 的恢复逻辑

修复：

- 新增 `SignalWatcher`
  - 专门等待 `SIGINT / SIGTERM / SIGHUP`
  - 收到信号后统一调用 `RequestPipelineStop()`
- 给 `FfmpegDemuxer` 增加 interrupt callback
  - 让阻塞中的 FFmpeg IO 能尽快返回
- `PipelineApp::Run()` 改成统一 `CleanupAndReturn()`
  - 等线程 `join`
  - 再释放模型 / 后处理
  - 最终让渲染线程自然析构 `DrmDisplay`

经验：

- 对 DRM/KMS 直显程序来说，`Ctrl+C` 不是“小事”
- 真正稳定的退出路径必须保证：
  - 线程先停
  - 显示后关
  - 最后再让进程结束

---

### 23. 不要去占系统 `Primary Plane`

现象：

- 即使程序能正常显示，退出不及时或清理不完整时，桌面可能一起黑掉

根因：

- 如果应用去抢 `Primary Plane` 或重新 modeset 主图层，
  那它和桌面 compositor 就在争同一层显示资源
- 一旦退出时机不好，残留的就是“主图层被盖住或被改坏”的状态

修复：

- 显示路径改成只申请 `Overlay Plane`
- 不再回退到会改动主图层的 legacy scanout
- 退出时只恢复 overlay 相关状态并释放 DRM Master

经验：

- 桌面环境里做视频叠加，应用最稳妥的策略是：
  - `Primary Plane` 留给系统
  - 应用只动 `Overlay Plane`

---

## 当前已经解决的问题

- RTSP 拉流阶段可定位
- `ffmpeg + rkmpp` 非紧凑 `NV12` 帧布局问题
- 纯 MPP 解码输出 `DMA-BUF`
- `decode -> RGA -> RKNN` 全 DMA-BUF 主链打通
- `RKNN io_mem(fd)` 跑通
- DRM/KMS 本地显示跑通
- DRM 显示从单缓冲优化到异步 page-flip
- DRM 显示已改成只使用 `Overlay Plane`，不再占用 `Primary Plane`
- DRM 显示缓冲加入 CPU/设备缓存同步
- `drmModePageFlip` 偶发失败可自恢复，不再一失败就永久黑屏
- `Ctrl+C / SIGTERM / SIGHUP` 已能触发受控退出，恢复显示状态并释放 DRM Master

---

## 当前仍需注意的限制

- 画框和文字仍然是 CPU 在 dumb buffer 映射内存上绘制
  - 功能没问题
  - 但不是“零 CPU 开销”

- 后处理仍然在 CPU

- 目前显示退出主要通过终端信号触发
  - 没有 `OpenCV q / ESC` 这种窗口交互
  - 但 `Ctrl+C` 现在已经会走受控清理路径

- `Overlay Plane` 模式要求系统本来就已经有一个激活中的桌面主图层
  - 如果当前 `CRTC` 上没有活动的 `Primary Plane`
  - 程序会直接进入无显示模式

- 显示链虽然已稳定很多，但框和文字仍然是 CPU 叠加，后续若追求更高显示吞吐，可继续考虑 RGA/plane 叠加

---

## 经验总结

1. `DMA-BUF` 改造最关键的是先明确“哪一段必须 zero-copy”
   本项目里真正重要的是：
   `decode output -> preprocess -> NPU input`

2. 不要把“压缩码流输入”也一起强行 DMA-BUF 化

3. `stride / offset / fd / size_with_stride` 比“像素格式名”更重要

4. 先跑通主链，再做显示

5. 显示链路的正确优化方向不是“更完整”，而是“更不阻塞”
