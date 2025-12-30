cmake_minimum_required(VERSION 3.10)
project(JetsonYolo_Project)

# --- 1. CẤU HÌNH CUDA 10.2 (Bắt buộc cho JetPack 4.6) ---
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-10.2)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-10.2/bin/nvcc)
include_directories(/usr/local/cuda-10.2/include)
link_directories(/usr/local/cuda-10.2/lib64)

# --- 2. CẤU HÌNH COMPILER ---
# Dùng C++14 để tương thích tốt nhất với CUDA 10.2
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
find_package(CUDA REQUIRED)

# Cờ tối ưu riêng cho GPU Maxwell (Jetson Nano)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_53,code=sm_53")

# --- 3. TÌM THƯ VIỆN ---
find_package(OpenCV REQUIRED)

# TensorRT Paths (Chuẩn JetPack 4.x)
include_directories(/usr/include/aarch64-linux-gnu)
link_directories(/usr/lib/aarch64-linux-gnu)

# Include folder dự án
include_directories(${CMAKE_SOURCE_DIR}/include)

# --- 4. SOURCE FILES ---
# Tự động tìm cả file .cpp và .cu (để lấy yololayer.cu)
file(GLOB SOURCES "src/*.cpp" "src/*.cu")

# --- 5. TẠO FILE THỰC THI ---
add_executable(${PROJECT_NAME} ${SOURCES})

# --- 6. LIÊN KẾT THƯ VIỆN ---
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    nvinfer
    nvinfer_plugin   # Bắt buộc để load plugin YOLO
    nvparsers
)

# --- 7. CỜ BIÊN DỊCH (ĐÃ FIX LỖI WALL) ---
target_compile_options(${PROJECT_NAME} PRIVATE
    # Cờ này chỉ dành cho C++ (GCC)
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -O3 -Wno-unused-function>
    
    # Cờ này chỉ dành cho CUDA (NVCC) - Không được dùng -Wall
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# Output directory
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

message(STATUS "Setup complete. Sources: ${SOURCES}")
