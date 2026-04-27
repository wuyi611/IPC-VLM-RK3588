cmake_minimum_required(VERSION 3.10)
project(rknn_yolov5_demo)

# 文件说明：
# 顶层构建脚本，负责组织源码、外部依赖和最终可执行程序的链接关系。

# 强制使用 C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TARGET_SOC "rk3588")

# ASAN 调试配置
if (ENABLE_ASAN)
    message(STATUS "BUILD WITH ADDRESS SANITIZER")
    set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif ()

# 包含子目录（utils、3rdparty）
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/utils/ utils.out)

# OpenCV + bundled FFmpeg-RK
find_package(OpenCV REQUIRED)

set(FFMPEG_RK_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ffmpeg-rk)
set(FFMPEG_RK_INCLUDE_DIR ${FFMPEG_RK_ROOT}/include)
set(FFMPEG_RK_LIB_DIR ${FFMPEG_RK_ROOT}/lib)
set(CURL_FALLBACK_INCLUDE_DIR /usr/include/x86_64-linux-gnu)
set(FFMPEG_RK_SHARED_LIBS
    ${FFMPEG_RK_LIB_DIR}/libavformat.so
    ${FFMPEG_RK_LIB_DIR}/libavcodec.so
    ${FFMPEG_RK_LIB_DIR}/libswresample.so
    ${FFMPEG_RK_LIB_DIR}/libavutil.so
)

foreach(FFMPEG_RK_LIB ${FFMPEG_RK_SHARED_LIBS})
    if (NOT EXISTS ${FFMPEG_RK_LIB})
        message(FATAL_ERROR "Missing bundled FFmpeg-RK library: ${FFMPEG_RK_LIB}")
    endif()
endforeach()

# 避免链接器把 FFmpeg 的本地依赖裁掉，确保运行时优先走 ffmpeg-rk 目录。
set(FFMPEG_RK_LINK_LIBS
    -Wl,--push-state,--no-as-needed
    ${FFMPEG_RK_SHARED_LIBS}
    -Wl,--pop-state
)

# RKNPU2
# set(LIBRKNNRT_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rknpu2/include)
# set(LIBRKNNRT ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rknpu2/Linux/librknnrt/aarch64/librknnrt.so)


# 源文件
add_executable(${PROJECT_NAME}
    src/app/app_options.cc
    src/app/main.cc
    src/app/pipeline_app.cc
    src/display/drm_display.cc
    src/media/ffmpeg_demuxer.cc
    src/media/mpp_decoder.cc
    src/cloud/providers/LLMApi.cpp
    src/model/postprocess.cc
    src/model/rknn_input_buffer_utils.cc
    src/model/yolov8_postprocess.cc
    src/pipeline/pipeline_utils.cc
    src/pipeline/stages/capture_stage.cc
    src/pipeline/stages/decode_stage.cc
    src/pipeline/stages/inference_stage.cc
    src/pipeline/stages/llm_stage.cc
    src/pipeline/stages/render_stage.cc
    src/model/yolov5_model.cc
    src/model/yolov8_model.cc
    3rdparty/allocator/dma/dma_alloc.cpp
)

# 头文件
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/allocator/dma
    /rk_sysroot/usr/include/libdrm
    ${FFMPEG_RK_INCLUDE_DIR}
    ${LIBRKNNRT_INCLUDES}
)

if (EXISTS ${CURL_FALLBACK_INCLUDE_DIR}/curl/curl.h)
    set_source_files_properties(
        src/cloud/providers/LLMApi.cpp
        PROPERTIES
        COMPILE_OPTIONS "-idirafter${CURL_FALLBACK_INCLUDE_DIR}"
    )
endif()

# 库文件
# 1. 显式告知链接器：去板子备份的库目录里搜
link_directories(/rk_sysroot/usr/lib/aarch64-linux-gnu)
link_directories(/rk_sysroot/lib/aarch64-linux-gnu)
link_directories(/rk_sysroot/usr/lib)
link_directories(/rk_sysroot/lib)

# 库链接
target_link_libraries(${PROJECT_NAME}
    imageutils
    imagedrawing
    fileutils
    audioutils
    ${FFMPEG_RK_LINK_LIBS}
    ${OpenCV_LIBS}    
    ${LIBRKNNRT}
    dl
    drm
    /rk_sysroot/usr/lib/aarch64-linux-gnu/libcurl.so.4
    /rk_sysroot/usr/lib/aarch64-linux-gnu/librockchip_mpp.so
    # 直接锁定物理真身，跳过所有软链接陷阱
    /rk_sysroot/usr/lib/aarch64-linux-gnu/blas/libblas.so.3.10.0
    /rk_sysroot/usr/lib/aarch64-linux-gnu/lapack/liblapack.so.3.10.0
    gfortran
)

# Linux 线程库支持
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_NAME} Threads::Threads)
endif()

# 设置运行时的库搜索路径
set_target_properties(${PROJECT_NAME} PROPERTIES
    BUILD_RPATH "${FFMPEG_RK_LIB_DIR}"
)
set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")

# # 安装配置 (执行 make install 时会将产物收集到一起)
# install(TARGETS ${PROJECT_NAME} DESTINATION .)
# # 修正：根据你的截图，model 目录似乎在工程根目录的 model/ 下
# if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/model")
#     install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/model DESTINATION .)
# endif()
