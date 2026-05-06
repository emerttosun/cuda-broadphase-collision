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
docs/                          # Project review and development roadmap
```

A long-form project review, current-state critique, parallelization
techniques and the literature-backed development roadmap (LBVH, spatial
hashing, sweep-and-prune, RT-core paths, profiling and presentation
guidelines) live in
[`docs/REVIEW_AND_ROADMAP.md`](docs/REVIEW_AND_ROADMAP.md).

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

Prerequisites:

- **Visual Studio 2022** (Community/Pro/Enterprise) with the "Desktop development with C++" workload — this provides MSVC, the Windows SDK, and MSBuild.
- **CMake 3.23 or newer** (`winget install Kitware.CMake` if you don't have it).
- **CUDA Toolkit 12.x** (Windows installer at https://developer.nvidia.com/cuda-toolkit-archive). Pick a version that matches your GPU: CUDA 13+ requires a Turing (sm_75) or newer GPU, while CUDA 12.x still supports Pascal (GTX 10-series) and Volta. The default preset targets CUDA 12.1 for the broadest GPU coverage.
- **A `raylib` source clone next to this project** (only needed for the visualizer):
  ```powershell
  cd ..
  git clone https://github.com/raysan5/raylib.git
  ```
  After cloning, the layout should be `parent/CMP674/` and `parent/raylib/` — the build system looks for raylib at `../raylib` automatically.

The build is driven by a `CMakePresets.json` file in the project root, so you do **not** need vcpkg, a toolchain file, or any custom `-D` flags. Two commands and you're done.

### Quick start

From the project root (`CMP674/`):

```powershell
cmake --preset default              # Configure (~20 s)
cmake --build --preset default      # Build benchmark + visualizer (~3-5 min first time)
```

That single configure step:

1. Auto-detects the `raylib` source clone next to the project.
2. Picks an installed CUDA 12.x toolkit on Windows (preset locks v12.1; see below to change).
3. Selects the matching Visual Studio CUDA toolset so MSBuild loads `CUDA <ver>.targets` correctly.
4. Sets `CMAKE_POLICY_VERSION_MINIMUM` so raylib's bundled GLFW (which still uses `cmake_minimum_required(VERSION 3.0)`) configures cleanly under modern CMake.
5. Defaults `BUILD_VISUALIZER=ON` because raylib was found.

After the build, the artifacts are:

```text
build\Release\collision_benchmark.exe
build\visualization\raylib-cuda-interop\Release\raylib_cuda_visualizer.exe
```

Run them:

```powershell
.\build\Release\collision_benchmark.exe
.\build\visualization\raylib-cuda-interop\Release\raylib_cuda_visualizer.exe
```

The CSV output is written to `results/timings.csv` relative to the working directory you ran the binary from (so the example above writes `build\Release\results\timings.csv`).

### Available presets

| Preset | What it does |
| --- | --- |
| `default` | Configure benchmark + visualizer, VS 2022 / x64 / CUDA 12.1 toolset. |
| `benchmark-only` | Same toolchain as `default` but skips the visualizer (no raylib needed). |

Use the second one with:

```powershell
cmake --preset benchmark-only
cmake --build --preset benchmark-only
```

### Customizing the preset

`CMakePresets.json` hardcodes the CUDA version and toolset path so the build "just works" on the development machine. You will likely need to edit it once for your own setup. The relevant fields are:

```json
"toolset": "cuda=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1",
"cacheVariables": {
    "CMAKE_CUDA_COMPILER": "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/bin/nvcc.exe"
}
```

Common edits:

- **Different CUDA version**: change both occurrences of `v12.1` to whatever you have installed (e.g. `v12.6`). Both fields must agree.
- **GPU compute capability**: by default the project compiles with `CMP674_CUDA_ARCHITECTURES=native`, which queries your local GPU at build time. If you want a fixed list (for portability or to skip an unsupported card), add to `cacheVariables`:
  ```json
  "CMP674_CUDA_ARCHITECTURES": "75;86;89"
  ```
  Use `61` for GTX 10-series, `75` for GTX 16xx / RTX 20xx, `86` for RTX 30xx, `89` for RTX 40xx.
- **raylib in a non-standard location**: add to `cacheVariables`:
  ```json
  "RAYLIB_SOURCE_DIR": "C:/path/to/raylib"
  ```
  Otherwise the project searches `../raylib`, `./raylib`, and `./external/raylib`.
- **Different Visual Studio**: change `"generator"` to `"Visual Studio 16 2019"` or whatever you have. The CUDA toolset path stays the same.

After editing the preset, re-run `cmake --preset default` (or whatever preset name) to apply changes; old build directories should be wiped first if the toolset or generator changed:

```powershell
Remove-Item -Recurse -Force build
cmake --preset default
cmake --build --preset default
```

### Building without presets

If you cannot use presets (CMake older than 3.23, or you want to script the flags yourself), the equivalent direct invocation is:

```powershell
cmake -S . -B build `
    -G "Visual Studio 17 2022" -A x64 `
    -T "cuda=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1" `
    "-DCMAKE_CUDA_COMPILER=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/bin/nvcc.exe" `
    -DBUILD_VISUALIZER=ON
cmake --build build --config Release
```

On Linux, WSL2, or Google Colab (no Visual Studio toolset, just nvcc on PATH):

```bash
cmake -S . -B build -DBUILD_VISUALIZER=ON
cmake --build build -j
./build/collision_benchmark
```

This produces both `build/collision_benchmark` and `build/visualization/raylib-cuda-interop/raylib_cuda_visualizer`. They share the same `cmp674_core` build artifacts, so nothing is recompiled twice.

### Build options

| Option | Default | Effect |
| --- | --- | --- |
| `BUILD_VISUALIZER` | `ON` if raylib source is found, otherwise `OFF` | Also build `raylib_cuda_visualizer`. Requires raylib and an interactive OpenGL/CUDA context. |
| `CMP674_CUDA_ARCHITECTURES` | `native` (CMake ≥ 3.24) or `60;61;70;75;80;86;89` | Which compute capabilities to generate code for. |
| `RAYLIB_SOURCE_DIR` | auto-detected at `../raylib` | Path to a raylib source clone, used as a CMake subproject. |

## Google Colab

Colab provides a CUDA-capable Tesla T4 (sm_75) and CUDA 12.x preinstalled, so the build is as simple as it gets — no preset needed, just plain CMake. The visualizer is OFF by default in this path because Colab has no display server.

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

Run it from the project root after a visualizer build:

```powershell
.\build\visualization\raylib-cuda-interop\Release\raylib_cuda_visualizer.exe 2500
```

The optional `2500` argument is the number of circles to simulate.

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
