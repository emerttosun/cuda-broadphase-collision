cmake_minimum_required(VERSION 3.20)

# This CMakeLists supports two invocation paths:
#   1. As a subdirectory of the root CMake project (preferred):
#        cmake -B build -DBUILD_VISUALIZER=ON
#      In this mode the parent project provides cmp674_core and all the
#      shared C/CUDA configuration.
#   2. Standalone, from inside this directory:
#        cmake -B build && cmake --build build
#      In this mode we re-enter the parent CMake just to build cmp674_core,
#      which keeps a single source of truth for the shared core.

if(NOT DEFINED PROJECT_NAME)
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
endif()

if(NOT TARGET cmp674_core)
    set(BUILD_VISUALIZER OFF CACHE BOOL "" FORCE)
    add_subdirectory(
        "${CMAKE_CURRENT_LIST_DIR}/../.."
        "${CMAKE_BINARY_DIR}/cmp674-core-from-visualizer"
    )
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
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(raylib_cuda_visualizer PRIVATE
    cmp674_core
    raylib
)

if(WIN32)
    target_link_libraries(raylib_cuda_visualizer PRIVATE winmm opengl32)
endif()

set_target_properties(raylib_cuda_visualizer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${CMP674_CUDA_ARCHITECTURES}"
)

target_compile_options(raylib_cuda_visualizer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
