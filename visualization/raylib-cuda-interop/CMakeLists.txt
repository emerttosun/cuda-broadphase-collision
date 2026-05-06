cmake_minimum_required(VERSION 3.24)

# Standalone-build path: when this directory is configured directly (not
# included by the root CMakeLists), bring in the parent project so
# cmp674_core, the CUDA setup and RAYLIB_SOURCE_DIR are all defined once.
if(NOT TARGET cmp674_core)
    project(RaylibCudaInteropVisualizer LANGUAGES C CUDA)
    set(BUILD_VISUALIZER OFF CACHE BOOL "" FORCE)
    add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../.."
                     "${CMAKE_BINARY_DIR}/cmp674-core-from-visualizer")
endif()

# raylib as a CMake subproject. Examples/games are skipped to keep
# configure/build time down.
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_GAMES    OFF CACHE BOOL "" FORCE)
add_subdirectory("${RAYLIB_SOURCE_DIR}"
                 "${CMAKE_BINARY_DIR}/raylib-build" EXCLUDE_FROM_ALL)

add_executable(raylib_cuda_visualizer
    src/main.c
    src/CudaSimulation.cu
)
target_include_directories(raylib_cuda_visualizer PRIVATE include)
target_link_libraries(raylib_cuda_visualizer PRIVATE cmp674_core raylib)
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
