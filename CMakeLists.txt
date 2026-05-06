cmake_minimum_required(VERSION 3.20)

# ---------------------------------------------------------------------------
# Pre-project setup: must run BEFORE project() so the CUDA language is
# initialized with the chosen toolkit, and so old subprojects (raylib's
# bundled glfw) inherit the policy compatibility shim.
# ---------------------------------------------------------------------------

# Auto-select a CUDA 12.x toolkit on Windows when none was specified. CUDA 13+
# dropped Pascal/Volta; 12.x supports the broad GTX 10-series .. RTX 40-series
# range that this project targets.
if(WIN32 AND NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
    foreach(_cuda_ver IN ITEMS v12.9 v12.8 v12.6 v12.5 v12.4 v12.3 v12.2 v12.1 v12.0)
        set(_cuda_root "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/${_cuda_ver}")
        if(EXISTS "${_cuda_root}/bin/nvcc.exe")
            set(CMAKE_CUDA_COMPILER "${_cuda_root}/bin/nvcc.exe"
                CACHE FILEPATH "Auto-selected CUDA compiler")
            # For the Visual Studio generator: pick the matching VS toolset so
            # MSBuild loads "CUDA <ver>.props" instead of the system default.
            set(CMAKE_VS_PLATFORM_TOOLSET_CUDA "${_cuda_root}"
                CACHE PATH "Auto-selected CUDA VS toolset")
            message(STATUS "CUDA toolkit auto-selected: ${_cuda_root}")
            break()
        endif()
    endforeach()
    unset(_cuda_root)
endif()

# raylib's bundled glfw declares `cmake_minimum_required(VERSION 3.0)`, which
# CMake 4.x rejects unless this compatibility shim is set.
if(NOT DEFINED CMAKE_POLICY_VERSION_MINIMUM)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.10
        CACHE STRING "Minimum policy version for legacy subprojects")
endif()

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

# ---------------------------------------------------------------------------
# raylib source auto-detection. The visualizer subproject expects either:
#   - RAYLIB_SOURCE_DIR set => raylib is added as a CMake subdirectory (no
#     install / find_package / vcpkg required), OR
#   - find_package(raylib) succeeds via CMAKE_PREFIX_PATH or vcpkg toolchain.
# We strongly prefer the first path: drop a raylib clone next to this project
# and everything builds with one cmake invocation.
# ---------------------------------------------------------------------------
if(NOT DEFINED RAYLIB_SOURCE_DIR)
    foreach(_p IN ITEMS
            "${CMAKE_CURRENT_SOURCE_DIR}/../raylib"
            "${CMAKE_CURRENT_SOURCE_DIR}/raylib"
            "${CMAKE_CURRENT_SOURCE_DIR}/external/raylib")
        if(EXISTS "${_p}/CMakeLists.txt" AND EXISTS "${_p}/src/raylib.h")
            get_filename_component(RAYLIB_SOURCE_DIR "${_p}" ABSOLUTE)
            message(STATUS "raylib source auto-detected: ${RAYLIB_SOURCE_DIR}")
            break()
        endif()
    endforeach()
    unset(_p)
endif()

# Default the visualizer ON when raylib is available, OFF otherwise.
if(DEFINED RAYLIB_SOURCE_DIR)
    option(BUILD_VISUALIZER "Build the raylib + CUDA-OpenGL interop visualizer" ON)
else()
    option(BUILD_VISUALIZER "Build the raylib + CUDA-OpenGL interop visualizer" OFF)
endif()

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

# ---------------------------------------------------------------------------
# Benchmark binary: orchestration + CSV writer that drives cmp674_core.
# ---------------------------------------------------------------------------
add_executable(collision_benchmark
    src/main.c
    src/Benchmark.c
)

target_link_libraries(collision_benchmark PRIVATE cmp674_core)

set_target_properties(collision_benchmark PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "${CMP674_CUDA_ARCHITECTURES}"
)

# ---------------------------------------------------------------------------
# Live visualizer (raylib + CUDA-OpenGL interop). Built by default whenever
# raylib source was found alongside the project; can be disabled with
# -DBUILD_VISUALIZER=OFF.
# ---------------------------------------------------------------------------
if(BUILD_VISUALIZER)
    if(NOT DEFINED RAYLIB_SOURCE_DIR)
        message(FATAL_ERROR
            "BUILD_VISUALIZER is ON but no raylib source was found. "
            "Either pass -DRAYLIB_SOURCE_DIR=<path> or place a raylib clone at "
            "one of: ../raylib, ./raylib, ./external/raylib (relative to ${CMAKE_CURRENT_SOURCE_DIR})."
        )
    endif()
    add_subdirectory(visualization/raylib-cuda-interop)
endif()
