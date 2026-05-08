cmake_minimum_required(VERSION 3.24)

# raylib's bundled glfw still declares cmake_minimum_required(VERSION 3.0);
# CMake 4.x rejects that unless this shim is set.
if(NOT DEFINED CMAKE_POLICY_VERSION_MINIMUM)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.10 CACHE STRING "")
endif()

# Auto-pick a CUDA 12.x toolkit on Windows if the preset/env didn't set one.
# CUDA 13+ dropped Pascal/Volta; 12.x covers GTX 10xx .. RTX 40xx.
if(WIN32 AND NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
    foreach(_v IN ITEMS v12.9 v12.8 v12.6 v12.5 v12.4 v12.3 v12.2 v12.1 v12.0)
        set(_root "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/${_v}")
        if(EXISTS "${_root}/bin/nvcc.exe")
            set(CMAKE_CUDA_COMPILER "${_root}/bin/nvcc.exe" CACHE FILEPATH "")
            set(CMAKE_VS_PLATFORM_TOOLSET_CUDA "${_root}" CACHE PATH "")
            break()
        endif()
    endforeach()
endif()

project(CudaCollisionDetection LANGUAGES C CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CUDA_STANDARD 17)

# CUDA arch. "native" = build for the local GPU. Override e.g. with
# -DCMP674_CUDA_ARCHITECTURES=86 or via CMakePresets cacheVariables.
# 61=GTX10xx, 75=GTX16/RTX20, 86=RTX30, 89=RTX40.
set(CMP674_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures")

option(BUILD_VISUALIZER "Build the raylib + CUDA-OpenGL interop visualizer" ON)

# raylib lives in-tree, always at this path. No search, no find_package.
set(RAYLIB_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/raylib")

# ---- cmp674_core: shared CPU/CUDA library, used by both binaries ----
add_library(cmp674_core STATIC
    src/Timer.c
    src/DataGenerator.c
    src/CpuCollision.c
    src/CudaBruteForce.cu
    src/CudaGrid.cu
    src/CudaHash.cu
    src/CudaLbvh.cu
)
target_include_directories(cmp674_core PUBLIC include)
set_target_properties(cmp674_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "${CMP674_CUDA_ARCHITECTURES}"
    POSITION_INDEPENDENT_CODE ON
)
target_compile_options(cmp674_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
if(UNIX)
    target_link_libraries(cmp674_core PUBLIC m)
endif()

# ---- collision_benchmark: CSV benchmark binary ----
add_executable(collision_benchmark src/main.c src/Benchmark.c)
target_link_libraries(collision_benchmark PRIVATE cmp674_core)
set_target_properties(collision_benchmark PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "${CMP674_CUDA_ARCHITECTURES}"
)

enable_testing()
add_executable(collision_parity_test tests/test_parity.c)
target_link_libraries(collision_parity_test PRIVATE cmp674_core)
set_target_properties(collision_parity_test PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "${CMP674_CUDA_ARCHITECTURES}"
)
add_test(NAME collision_parity_test COMMAND collision_parity_test)

# ---- raylib_cuda_visualizer: optional live demo ----
if(BUILD_VISUALIZER)
    if(NOT EXISTS "${RAYLIB_SOURCE_DIR}/src/raylib.h")
        message(FATAL_ERROR
            "raylib source not found at ${RAYLIB_SOURCE_DIR}\n"
            "Run from the project root:\n"
            "    git clone https://github.com/raysan5/raylib.git external/raylib")
    endif()
    add_subdirectory(visualization/raylib-cuda-interop)
endif()
