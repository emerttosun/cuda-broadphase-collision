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

## macOS Note

macOS is used only for writing code, Git management, README editing, and report preparation. CUDA compilation and benchmark execution are not expected to run on macOS. Run the benchmark on Linux, Windows, WSL2, or Google Colab with an NVIDIA GPU.

## Build With CMake

On a CUDA-capable machine:

```bash
mkdir build
cd build
cmake ..
cmake --build .
./collision_benchmark
```

The CSV output is written to:

```text
results/timings.csv
```

If you run the binary from the `build` directory, the output path will be `build/results/timings.csv`.

## Manual nvcc Build

If CMake is not available or gives issues, build manually from the project root:

```bash
nvcc src/CudaBruteForce.cu src/CudaGrid.cu \
src/main.c src/Timer.c src/CpuCollision.c src/DataGenerator.c src/Benchmark.c \
-Iinclude -std=c++17 -o collision_benchmark

./collision_benchmark
```

If building from inside `build`, use:

```bash
nvcc ../src/CudaBruteForce.cu ../src/CudaGrid.cu \
../src/main.c ../src/Timer.c ../src/CpuCollision.c ../src/DataGenerator.c ../src/Benchmark.c \
-I../include -std=c++17 -o collision_benchmark
```

## Google Colab

```python
!nvidia-smi
!nvcc --version
!git clone git@github.com:emerttosun/CMP674.git collision-cuda-project
%cd collision-cuda-project
!mkdir -p build
%cd build
!cmake ..
!make -j2
!./collision_benchmark
```

If SSH clone is not configured in Colab, use the HTTPS URL instead:

```python
!git clone https://github.com/emerttosun/CMP674.git collision-cuda-project
```

If CMake fails in Colab, use manual `nvcc`:

```python
%cd /content/collision-cuda-project
!nvcc src/CudaBruteForce.cu src/CudaGrid.cu \
src/main.c src/Timer.c src/CpuCollision.c src/DataGenerator.c src/Benchmark.c \
-Iinclude -std=c++17 -o collision_benchmark
!./collision_benchmark
```

## Live Raylib CUDA Visualization

The core benchmark writes CSV results. A separate live visualizer is available under:

```text
visualization/raylib-cuda-interop
```

This target uses raylib for the window and UI overlay, while CUDA maps an OpenGL VBO and writes particle positions/colors directly into it through CUDA-OpenGL interop. It is intended for a Windows machine with an NVIDIA GPU.

See [visualization/raylib-cuda-interop/README.md](visualization/raylib-cuda-interop/README.md) for build instructions.

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

The default maximum radius is `2.0`, and the smallest tested grid cell size is `5.0`. This keeps the 8-neighbor grid search valid for the default configuration because the maximum collision distance is smaller than the smallest cell size.
