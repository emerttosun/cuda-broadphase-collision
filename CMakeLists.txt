cmake_minimum_required(VERSION 3.18)

project(CudaCollisionBenchmark LANGUAGES C CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMP674_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures for CMP674 targets")
else()
    set(CMP674_CUDA_ARCHITECTURES "60;61;70;75;80;86;89" CACHE STRING "CUDA architectures for CMP674 targets")
endif()

add_executable(collision_benchmark
    src/main.c
    src/Timer.c
    src/DataGenerator.c
    src/CpuCollision.c
    src/CudaBruteForce.cu
    src/CudaGrid.cu
    src/Benchmark.c
)

target_include_directories(collision_benchmark PRIVATE include)

set_target_properties(collision_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${CMP674_CUDA_ARCHITECTURES}"
)

target_compile_options(collision_benchmark PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

if(UNIX)
    target_link_libraries(collision_benchmark PRIVATE m)
endif()
