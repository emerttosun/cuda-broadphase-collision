cmake_minimum_required(VERSION 3.18)

project(CudaCollisionBenchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMP674_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures for CMP674 targets")
else()
    set(CMP674_CUDA_ARCHITECTURES "60;61;70;75;80;86;89" CACHE STRING "CUDA architectures for CMP674 targets")
endif()

# Fetch Raylib
include(FetchContent)
FetchContent_Declare(
    raylib
    URL https://github.com/raysan5/raylib/archive/refs/tags/5.0.tar.gz
)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_GAMES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(raylib)

add_executable(collision_benchmark
    src/main.cpp
    src/Application.cu
    src/Benchmark.cpp
    src/DataGenerator.cpp
    src/CpuBruteForceDetector.cpp
    src/CudaBruteForceDetector.cu
    src/CudaGridDetector.cu
    src/CudaLBVHDetector.cu
)

target_include_directories(collision_benchmark PRIVATE include)

set_target_properties(collision_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${CMP674_CUDA_ARCHITECTURES}"
)

target_compile_options(collision_benchmark PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

target_link_libraries(collision_benchmark PRIVATE raylib)

if(UNIX)
    target_link_libraries(collision_benchmark PRIVATE m)
endif()
