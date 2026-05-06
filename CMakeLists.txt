cmake_minimum_required(VERSION 3.20)

project(CudaCollisionDetection LANGUAGES C CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMP674_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures for CMP674 targets")
else()
    set(CMP674_CUDA_ARCHITECTURES "60;61;70;75;80;86;89" CACHE STRING "CUDA architectures for CMP674 targets")
endif()

option(BUILD_VISUALIZER "Build the raylib + CUDA-OpenGL interop visualizer" OFF)

# ---------------------------------------------------------------------------
# Shared core: every CPU/CUDA broadphase implementation plus the helpers
# (Timer, RNG via header, DataGenerator). Both the benchmark binary and the
# live visualizer link against this static library, so include paths and
# compile flags only need to be configured once.
# ---------------------------------------------------------------------------
add_library(cmp674_core STATIC
    src/Timer.c
    src/DataGenerator.c
    src/CpuCollision.c
    src/CudaBruteForce.cu
    src/CudaGrid.cu
)

target_include_directories(cmp674_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(cmp674_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${CMP674_CUDA_ARCHITECTURES}"
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(cmp674_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

if(UNIX)
    target_link_libraries(cmp674_core PUBLIC m)
endif()

# ---------------------------------------------------------------------------
# Benchmark binary: orchestration + CSV writer that drives cmp674_core.
# ---------------------------------------------------------------------------
add_executable(collision_benchmark
    src/main.c
    src/Benchmark.c
)

target_link_libraries(collision_benchmark PRIVATE cmp674_core)

# ---------------------------------------------------------------------------
# Optional live visualizer (raylib + CUDA-OpenGL interop). Off by default
# because it requires raylib and an interactive GPU/OpenGL context.
# ---------------------------------------------------------------------------
if(BUILD_VISUALIZER)
    add_subdirectory(visualization/raylib-cuda-interop)
endif()
