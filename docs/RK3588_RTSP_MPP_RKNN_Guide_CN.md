# RK3588 RTSP / MPP / RKNN 使用说明

## 最终架构

当前程序的主链路如下：

1. `FFmpeg`
   负责 `RTSP` 拉流与解复用。

2. `MPP`
   负责视频硬件解码，输出 `YUV420SP` 解码帧。

3. `RGA`
   负责推理前预处理：
   - 缩放
   - letterbox
   - `YUV -> RGB`

4. `RKNN / rknpu2`
   负责 `YOLOv5` 模型推理。

5. `CPU 后处理`
   负责输出解码、阈值筛选、NMS。

6. `DRM/KMS`
   负责本地显示：
   - 只申请一个 `Overlay Plane`
   - 不占用系统 `Primary Plane`
   - 检测框和文字当前仍由 CPU 画到 overlay buffer

## 哪些部分是硬件加速

- `MPP`：硬件视频解码
- `RGA`：预处理
- `RKNN / NPU`：模型推理

## 哪些部分是 CPU

- `FFmpeg`：RTSP 会话、网络收包、demux
- `YOLOv5 后处理`
- OSD 画框与文字叠加

## 统计字段说明

程序现在输出的统计项为：

- `decode`
  从压缩包进入系统到解码帧可用的平均耗时。

- `wait_infer`
  解码完成后，在进入推理前的平均等待时间。

- `pre`
  预处理平均耗时。

- `npu`
  `rknn_run` 平均耗时。

- `post`
  后处理平均耗时。

- `infer`
  推理阶段总耗时，约等于：
  `pre + npu + post`

- `wait_render`
  推理完成后，在渲染线程消费前的平均等待时间。

- `latency`
  从一帧进入系统到最终被显示/统计线程消费的总延迟。

## 推荐运行参数

### 低延迟显示

```bash
./rknn_yolov5_demo model/yolov5s_relu_i8_det.rknn "rtsp://admin:password@camera_ip:554/Streaming/Channels/101" \
  --labels ./labels.txt \
  --infer-threads 1 \
  --capture-queue 1 \
  --decode-queue 1 \
  --render-queue 1 \
  --stats 30
```

### 纯检测吞吐

```bash
./rknn_yolov5_demo model/yolov5s_relu_i8_det.rknn "rtsp://admin:password@camera_ip:554/Streaming/Channels/101" \
  --labels ./labels.txt \
  --infer-threads 1 \
  --capture-queue 1 \
  --decode-queue 1 \
  --render-queue 1 \
  --stats 30 \
  --no-display
```

## 参数建议

- 对 25fps 左右的相机流，通常优先尝试 `--infer-threads 1`
  单线程推理如果已经能覆盖源流帧率，多开线程不会带来收益，反而可能增加调度开销和延迟。

- 队列优先用 `1 / 1 / 1`
  这是偏低延迟配置。

- 如果追求吞吐而不是显示，建议使用 `--no-display`
  这样可以避免显示路径的额外开销。

- 当前显示路径统一使用 `DRM/KMS`
  用户可通过 `Ctrl+C` 退出；程序会尽量等待各线程收尾，并恢复显示状态、释放 DRM Master。

- 当前显示模式依赖系统桌面已经激活主图层
  程序本身只叠加一个 `Overlay Plane`，不会主动接管整个屏幕。
