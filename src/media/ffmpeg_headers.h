#ifndef RKNN_YOLOV5_DEMO_MEDIA_FFMPEG_HEADERS_H_
#define RKNN_YOLOV5_DEMO_MEDIA_FFMPEG_HEADERS_H_

// 文件说明：
// 统一封装 FFmpeg 头文件的 C 链接方式，避免在多个 C++ 文件里重复书写 `extern "C"`。

/*
 * FFmpeg 的 Debian/Ubuntu 头文件本身不总是替 C++ 调用方自动包好 `extern "C"`。
 * 如果直接在 C++ 工程里包含这些头，编译器会把 av_* 函数按 C++ 符号处理，
 * 最终在链接阶段表现为：
 *   undefined reference to `avformat_open_input(...)`
 *   undefined reference to `av_packet_unref(...)`
 *
 * 因此这里统一提供一个“唯一合法入口”来包含 FFmpeg 头文件，
 * 避免某个 cpp / h 文件遗漏 `extern "C"` 后再次引入同类问题。
 */

#ifdef __cplusplus
// 统一以 C 链接方式引入 FFmpeg 的公开头文件。
extern "C" {
#endif

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/dict.h>
#include <libavutil/error.h>
#include <libavutil/imgutils.h>

#ifdef __cplusplus
}
#endif

#endif  // RKNN_YOLOV5_DEMO_MEDIA_FFMPEG_HEADERS_H_
