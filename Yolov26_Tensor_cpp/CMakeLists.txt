cmake_minimum_required(VERSION 3.10)
project(YOLO26_TensorRT)

# 1. 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 2. 寻找依赖库
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# 3. 设置头文件路径（修改这一部分）
include_directories(${OpenCV_INCLUDE_DIRS})  # 这会自动包含 /usr/include/opencv4
include_directories(/usr/local/cuda/include)
include_directories(/usr/include/aarch64-linux-gnu)  # TensorRT 头文件通常在这里

# 4. 设置库文件路径
link_directories(/usr/lib/aarch64-linux-gnu)
link_directories(/usr/local/cuda/lib64)

# 5. 生成可执行文件
add_executable(yolo26_app main.cpp yolo.cpp)
# 添加 USB 摄像头版本的可执行文件
add_executable(yolo26_usb main_usb.cpp yolo.cpp)
add_executable(yolo26_csi main_csi.cpp yolo.cpp)

# 6. 链接库
target_link_libraries(yolo26_app ${OpenCV_LIBS} ${CUDA_LIBRARIES} nvinfer nvonnxparser cudart)
# 链接库 (保持和你之前的一样)
target_link_libraries(yolo26_usb 
    ${OpenCV_LIBS} 
    ${CUDA_LIBRARIES} 
    nvinfer 
    nvonnxparser
    cudart
)
target_link_libraries(yolo26_csi
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    nvinfer
    nvonnxparser
    cudart
)