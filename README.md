# CUDA-Based Broad-Phase Collision Detection with CPU-GPU Performance Analysis

This project establishes a literature-backed, rigorous comparison between CPU-integrated and GPU-integrated broad-phase collision detection methodologies. It demonstrates massive performance improvements using advanced CUDA optimizations and data structures.

The application functions as a **single, unified executable** that serves both as a visual presentation demo (using Raylib with OpenGL-CUDA interop) and a headless benchmarking suite.

## Core Architectural Features

The project is built around a highly optimized C++ Object-Oriented design, featuring:

- **Structure of Arrays (SoA)**: The data layout (`CirclesSoA`) guarantees mathematically coalesced memory access across all CUDA threads, maximizing memory bandwidth.
- **Unified Interfaces**: All collision methods inherit from `ICollisionDetector` and use a unified `CollisionMath` core containing `__device__ __host__` inline mathematics to ensure exact parity between CPU and GPU routines.
- **Transparent Benchmarking**: Results strictly isolate Host-to-Device/Device-to-Host memory transfer times from raw kernel execution times.

## Implemented Methods

### 1. CPU Brute Force (Baseline)
The CPU method checks every object pair using nested loops (`O(n^2)` complexity). It is used as the foundational baseline to calculate exact speedups.

### 2. CUDA Brute Force (Shared Memory Tiled)
The baseline GPU approach distributes work across GPU threads but incorporates **Shared Memory Tiling**. By cooperatively loading circle data into `__shared__` memory at the block level (a classic N-body optimization), global memory reads are drastically reduced. **CUDA Streams** are used to overlap memory transfers with execution.

### 3. CUDA Uniform Grid
The grid method divides the scene into square cells:
1. Compute a `cellId` for every circle.
2. Sort objects by `cellId` using `thrust::sort_by_key`.
3. Build cell start/end boundaries.
4. For each object, compare only objects in its own cell and the 8 neighboring cells.

### 4. CUDA Linear Bounding Volume Hierarchy (LBVH)
The flagship advanced method based on Karras (2012). This implementation includes:
- **Morton Code Generation**: Z-order curve mapping via 30-bit Morton codes generated from AABB centroids.
- **Radix Sorting**: Leveraging Thrust (which wraps CUB's highly optimized radix sort) to arrange elements spatially.
- **Hierarchy Construction**: A custom CUDA kernel performs bottom-up tree construction, finding split points via binary search over the Morton codes.
- **Parallel AABB Traversal**: Each thread traverses the tree independently using a thread-local stack to compute collision pairs efficiently.

## Single Unified Executable

### Live Interactive Visualizer (Demo Mode)
Running the executable normally launches the Raylib visualizer.
- Press **1, 2, 3, or 4** to instantly toggle between CPU Brute Force, CUDA Brute Force, CUDA Grid, and CUDA LBVH.
- **Zero-Copy Advantage**: When a GPU method is active, the application uses **OpenGL-CUDA interop** to map the VBO directly to CUDA memory space. A custom SoA VBO-writer kernel populates vertex data immediately on the device, bypassing the CPU entirely.
- Press **SPACE** to pause and **R** to reset distributions (Uniform vs. Clustered).

### Headless Benchmark Mode
Running the executable with the `--benchmark` flag launches the headless CSV generator. It tests multiple object counts across different distributions.

```bash
./collision_benchmark --benchmark
```
The CSV output is written to `results/timings.csv`.

## Build With CMake

The project uses CMake and includes Raylib as a Git submodule.

On a CUDA-capable machine (Linux, Windows, WSL2, or Colab):

```bash
# Clone the repository with submodules
git clone --recursive https://github.com/emerttosun/CMP674.git
cd CMP674

mkdir build
# cd build
cmake -B build
cmake --build build --config Release
```

To run the interactive visualizer (Requires a Display):
```bash
./build/collision_benchmark
```

To run the headless benchmark (Perfect for Colab/SSH):
```bash
./build/collision_benchmark --benchmark
```

## Google Colab Instructions

You can run the benchmark headless on Google Colab to gather data using a high-end GPU.

```python
!nvidia-smi
!nvcc --version
!git clone --recursive https://github.com/emerttosun/CMP674.git collision-cuda-project
%cd collision-cuda-project
!cmake -B build
!cmake --build build -j2
!./build/collision_benchmark --benchmark
```

## CSV Columns

`results/timings.csv` provides an academic-grade performance matrix:

- `object_count`
- `distribution_type`
- `method_name`
- `collision_count`
- `candidate_pair_count`
- `total_time_ms` (End-to-end time)
- `memory_transfer_time_ms` (Time spent transferring data H2D and D2H)
- `kernel_execution_time_ms` (Pure GPU execution time)
- `speedup_vs_cpu`
- `grid_cell_size` (Only for Grid method)
- `max_objects_in_cell` (Only for Grid method)
- `avg_objects_per_non_empty_cell` (Only for Grid method)
- `dense_cell_count` (Only for Grid method)
