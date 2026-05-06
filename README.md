# CUDA-Based Broad-Phase Collision Detection with CPU-GPU Performance Analysis

This project compares CPU and GPU approaches for 2D circle collision detection. It includes a CPU brute force baseline, a CUDA brute force kernel, and a CUDA uniform grid broad-phase method. The benchmark runs both uniform and clustered data distributions and writes measurements to `results/timings.csv`.

The code is intentionally written in a C-style structure. CPU-side files use C, while CUDA files are compiled by `nvcc`. The uniform grid implementation uses Thrust `sort_by_key`, so those `.cu` files are CUDA C++ internally even though the public API and data structures are C-style.

## Collision Detection

Collision detection checks whether two objects intersect. In this project every object is a 2D circle with:

- `x`
- `y`
- `radius`

Two circles collide when:

```c
dx = x1 - x2;
dy = y1 - y2;
distanceSquared = dx * dx + dy * dy;
collision = distanceSquared <= (r1 + r2) * (r1 + r2);
```

The check itself lives in `include/CollisionMath.h` as a `__device__ __host__ inline` function, so the CPU baseline, the CUDA brute force kernel, the CUDA uniform-grid kernel and the live visualizer all use exactly the same formula.

## Broad Phase

Broad phase collision detection reduces the number of object pairs that need detailed testing. Instead of comparing every pair in the scene, a spatial structure first produces a smaller set of candidate pairs. This project uses a uniform grid broad phase.

## Implemented Methods

### CPU Brute Force

The CPU method checks every object pair using nested loops. It is simple and exact, but it has `O(n^2)` complexity. It is used as the baseline for speedup calculations.

### CUDA Brute Force

The CUDA brute force method keeps the same all-pairs logic but distributes work across GPU threads. Each thread handles one object and compares it with later objects. Collision and candidate counts are accumulated with `atomicAdd`.

### CUDA Uniform Grid

The CUDA grid method divides the scene into square cells:

1. Compute a `cellId` for every circle.
2. Sort objects by `cellId` using Thrust `sort_by_key`.
3. Build `cellStart` and `cellEnd` arrays.
4. For each object, compare only objects in its own cell and the 8 neighboring cells.

The method reports both collision count and candidate pair count. Candidate pair count is useful because it shows how much work the broad phase generated.

## Data Distributions

### Uniform

Objects are placed randomly across the scene. This usually produces balanced grid occupancy and helps the uniform grid reduce candidate pairs effectively.

### Clustered

Objects are generated around 3 or 4 cluster centers. Clustered data can overload a few grid cells, which increases candidate pair count and reduces grid efficiency.

## Dense Cell Metrics

For clustered analysis, the grid reports:

- `max_objects_in_cell`: largest object count found in any grid cell.
- `avg_objects_per_non_empty_cell`: average occupancy among cells that contain at least one object.
- `dense_cell_count`: number of cells whose occupancy is at least the dense threshold.

The default dense cell threshold is `128`. Dense cell optimization is not implemented in this first version; only the metrics are reported.

## Project Layout

```
include/                       # Public headers (data types + algorithms)
src/                           # Shared core implementation + benchmark binary
visualization/raylib-cuda-interop/   # Optional live visualizer (raylib + CUDA)
```

The build is driven by a single root `CMakeLists.txt` that produces:

- `cmp674_core` — a CUDA-aware static library that contains every shared
  algorithm and helper (`Timer`, `DataGenerator`, `CpuCollision`,
  `CudaBruteForce`, `CudaGrid`). Its public include directory is `include/`,
  so any target that links against `cmp674_core` automatically sees the
  shared headers.
- `collision_benchmark` — the CSV benchmark binary. It only contains
  `src/main.c` and `src/Benchmark.c` (the orchestration that drives the
  `BroadphaseMethod` table) and links against `cmp674_core`.
- `raylib_cuda_visualizer` — the optional live demo. Built only when
  `-DBUILD_VISUALIZER=ON` is passed. Links against `cmp674_core` and
  `raylib`.

This means the include path, CUDA architecture list, separable-compilation
setting and `--expt-relaxed-constexpr` flag are configured exactly once on
`cmp674_core` and inherited by every consumer.

## macOS Note

macOS is used only for writing code, Git management, README editing, and report preparation. CUDA compilation and benchmark execution are not expected to run on macOS. Run the benchmark on Linux, Windows, WSL2, or Google Colab with an NVIDIA GPU.

## Build With CMake

On a CUDA-capable machine:

```bash
cmake -B build
cmake --build build -j
./build/collision_benchmark
```

The CSV output is written to `results/timings.csv` relative to the working
directory you run the binary from (so the example above writes
`build/results/timings.csv`).

### Build options

| Option | Default | Effect |
| --- | --- | --- |
| `BUILD_VISUALIZER` | `OFF` | Also build `raylib_cuda_visualizer`. Requires raylib and an interactive OpenGL/CUDA context. |
| `CMP674_CUDA_ARCHITECTURES` | `native` (CMake ≥ 3.24) or `60;61;70;75;80;86;89` | Which compute capabilities to generate code for. |

### Benchmark + visualizer in one configure

```bash
cmake -B build -DBUILD_VISUALIZER=ON \
    -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build -j
```

This produces both `build/collision_benchmark` and
`build/visualization/raylib-cuda-interop/raylib_cuda_visualizer`. The
visualizer reuses the same `cmp674_core` build artifacts as the benchmark, so
nothing is recompiled twice.

### Visualizer-only build

The visualizer's CMakeLists also works standalone for backwards
compatibility:

```bash
cd visualization/raylib-cuda-interop
cmake -B build -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build
```

In standalone mode the visualizer pulls in the parent project just to build
`cmp674_core`; you still get the same shared library underneath.

## Google Colab

```python
!nvidia-smi
!nvcc --version
!git clone https://github.com/emerttosun/cuda-broadphase-collision.git collision-cuda-project
%cd collision-cuda-project
!cmake -B build
!cmake --build build -j2
!./build/collision_benchmark
```

If you want the live visualizer in Colab, add `-DBUILD_VISUALIZER=ON` and
make sure raylib is installed; raylib needs an X server, so Colab is not the
typical environment for it.

## Live Raylib CUDA Visualization

The core benchmark writes CSV results. The optional live visualizer lives at:

```text
visualization/raylib-cuda-interop
```

It uses raylib for the window and UI overlay while CUDA maps an OpenGL VBO
and writes particle vertices directly into it through CUDA-OpenGL interop.
It links against the same `cmp674_core` library as the benchmark, so the
collision math is shared.

See [visualization/raylib-cuda-interop/README.md](visualization/raylib-cuda-interop/README.md) for build instructions, raylib install steps and the in-app keyboard controls.

## CSV Columns

`results/timings.csv` contains:

- `object_count`
- `distribution_type`
- `method_name`
- `collision_count`
- `candidate_pair_count`
- `kernel_time_ms` — pure kernel/loop time. CUDA rows exclude H2D/D2H transfers; CPU row equals total time.
- `total_time_ms` — wall-clock time including allocations and host-device transfers. CPU row equals kernel time.
- `speedup_vs_cpu` — `CPU total_time_ms / method total_time_ms`. For CPU rows it is `1.0`.
- `grid_cell_size`
- `max_objects_in_cell`
- `avg_objects_per_non_empty_cell`
- `dense_cell_count`

The CPU baseline is timed with `clock_gettime(CLOCK_MONOTONIC)` (or
`QueryPerformanceCounter` on Windows), so the CPU number is wall-clock time and
directly comparable to the CUDA `total_time_ms`.

## Expected Results

CUDA brute force should usually be faster than CPU brute force for larger object counts, but it still performs all-pairs work. CUDA uniform grid should reduce candidate pair count strongly for uniform data. In clustered data, some cells can become dense, causing higher candidate counts and weaker speedup.

Small collision count differences can happen if floating point behavior differs across CPU and GPU hardware. The benchmark uses the same circle formula for all methods, so results should normally be very close.

The default maximum radius is `2.0`, and the smallest tested grid cell size is `5.0`. This keeps the 8-neighbor grid search valid for the default configuration because the maximum collision distance is smaller than the smallest cell size. If `cell_size < 2 * max_radius`, `run_cuda_uniform_grid` prints a warning to `stderr` because the 9-cell neighborhood may then miss collisions.
