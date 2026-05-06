cmake_minimum_required(VERSION 3.20)

project(RaylibCudaInteropVisualizer LANGUAGES C CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMP674_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures for CMP674 targets")
else()
    set(CMP674_CUDA_ARCHITECTURES "60;61;70;75;80;86;89" CACHE STRING "CUDA architectures for CMP674 targets")
endif()

if(DEFINED RAYLIB_SOURCE_DIR)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_GAMES OFF CACHE BOOL "" FORCE)
    add_subdirectory("${RAYLIB_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/raylib-build" EXCLUDE_FROM_ALL)
else()
    find_package(raylib CONFIG REQUIRED)
endif()

add_executable(raylib_cuda_visualizer
    src/main.c
    src/CudaSimulation.cu
)

target_include_directories(raylib_cuda_visualizer PRIVATE
    include
    ../../include
)

target_link_libraries(raylib_cuda_visualizer PRIVATE raylib)

set_target_properties(raylib_cuda_visualizer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${CMP674_CUDA_ARCHITECTURES}"
)

target_compile_options(raylib_cuda_visualizer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
