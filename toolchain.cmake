# 1. 基础环境定义
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# 2. 编译器定义 (显式指向 /usr/bin，跳过 SYSROOT 偏移)
# 这两个是 x86 原生程序，运行速度极快
set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)

# 3. 系统根目录 (只用于找库和头文件)
set(CMAKE_SYSROOT /rk_sysroot)
set(CMAKE_FIND_ROOT_PATH /rk_sysroot)

# 4. 强制指定依赖路径
set(OpenCV_DIR /rk_sysroot/usr/lib/aarch64-linux-gnu/cmake/opencv4)

# 5. 搜索模式：关键设置
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER) # 核心：从不从 SYSROOT 里找编译器/构建程序
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)  # 只从 SYSROOT 找库
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)  # 只从 SYSROOT 找头文件
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)  # 只从 SYSROOT 找配置文件

# 6. 告诉 CMake 编译器是正常的，跳过简单的编译测试（可选，若还是 broken 则开启）
set(CMAKE_C_COMPILER_WORKS 1)
set(CMAKE_CXX_COMPILER_WORKS 1)