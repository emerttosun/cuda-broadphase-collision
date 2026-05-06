# Proje İncelemesi: CUDA Broad-Phase Collision Detection + Raylib Visualizer

## Context

Bu proje iki ayrı parçadan oluşuyor:

1. **Çekirdek benchmark** (`src/`, `include/`): 2D daire çarpışma tespiti için 3 yöntem
   karşılaştırılıyor — CPU brute force (referans), CUDA brute force, CUDA uniform
   grid (broad-phase). Sonuçlar `results/timings.csv` dosyasına yazılıyor.
2. **Canlı görselleştirme** (`visualization/raylib-cuda-interop/`): raylib pencere
   açıyor, rlgl ile bir VBO üretiyor, CUDA bu VBO'yu `cudaGraphicsGLRegisterBuffer`
   ile maple'iyor ve vertex'leri doğrudan GPU üzerinde yazıyor. Yani particle
   verisi için CPU mirroring **yok** — interop doğru kurulmuş.

Hedef: tüm proje saf C + CUDA, modüler ve generic yapı. Bu hedef bazı yerlerde
karşılanmıyor. Aşağıda hatalar, tutarsızlıklar ve hedeflere göre sapmalar
listelenmiştir.

---

## 1. "CPU Mirroring var mı?" — gerçek durum

Görselleştirmenin asıl veri akışı:

- `g_balls` (DeviceBall*) → tamamen GPU bellekte (`cudaMalloc`).
- VBO → `rlLoadVertexBuffer(NULL, size, dynamic=true)` ile boş açılıyor, asla
  CPU'dan beslenmiyor.
- `cudaGraphicsGLRegisterBuffer` → VBO interop için kayıt ediliyor.
- Her frame: `cudaGraphicsMapResources` → `write_vbo_kernel` doğrudan VBO'ya
  yazıyor → `cudaGraphicsUnmapResources` → raylib/rlgl aynı VBO'yu üçgen olarak
  çiziyor.

**Particle verisi için CPU mirroring yok.** Sadece `collision_count`,
`candidate_pair_count`, `gpu_time_ms` (toplam 16+4 byte) overlay metni için
host'a kopyalanıyor.

CPU mirroring **gerçekten** olan tek yer raylib'in kendi UI overlay rendering'i:
`DrawRectangle`, `DrawText`, `BeginDrawing/EndDrawing` zinciri raylib'in batch
sistemini kullanıyor (CPU'da vertex toplanıp her frame GPU'ya yükleniyor). Bu
raylib'in çalışma modeline ait, particle'larla ilgili değil. Tüm UI'yi bile
GPU'ya almak isteniyorsa ayrı bir custom text rendering yazmak gerekir.

---

## 2. Saf C hedefini kıran yerler

| # | Yer | Sorun |
|---|---|---|
| 2.1 | `visualization/raylib-cuda-interop/src/main.cpp` | Dosya **C++**. `<cstdio>`, `<cstdlib>`, `<cstring>`, `std::atoi`, `std::fprintf`, `std::memset`, ve `Color{...}` C++ uniform initialization kullanıyor. `main.c` olmalı; raylib zaten C API. |
| 2.2 | `visualization/raylib-cuda-interop/CMakeLists.txt:3` | `LANGUAGES CXX CUDA` — main C'ye taşınınca `C CUDA` olmalı. |
| 2.3 | `src/main.cu` | Sıfır CUDA içeriği var ama `.cu` uzantılı. Aslında saf C; kök benchmark için `main.c` olmalı, link aşamasında `.cu` nesneleriyle birleşir. |
| 2.4 | `visualization/.../CudaSimulation.cu:7` | `#include <windows.h>` ve `<GL/gl.h>` doğrudan dahil ediliyor — Linux/macOS taşınabilirliği kırılır. raylib zaten platforma göre OpenGL header'ını dahil ediyor; bu satırlar gereksiz. |

---

## 3. Modüler/Generic hedefini kıran yerler

| # | Yer | Sorun |
|---|---|---|
| 3.1 | `src/CudaBruteForce.cu:6` ve `visualization/.../CudaSimulation.cu:127-138` | `device_circles_collide` mantığı (dx*dx+dy*dy ≤ (r1+r2)^2) en az 3 yerde kopyalanmış (CPU, CUDA brute, CUDA grid, ek olarak görselleştirici). Tek bir `__device__ __host__ inline` header'a (örn. `include/CollisionMath.h`) taşınmalı. |
| 3.2 | `src/DataGenerator.c:6-13` ve `visualization/.../CudaSimulation.cu:34-42` | `lcg_next` + `random_float` aynı LCG, hem host'ta hem device'ta yeniden yazılmış. Ortak bir `__device__ __host__` RNG header'ı yapılabilir. |
| 3.3 | `visualization/.../CudaSimulation.cu:62-72` | Cluster merkezleri **hard-coded** (`width*0.25f`, `height*0.30f` ...). Halbuki `src/DataGenerator.c:55-59` rastgele cluster merkezi üretiyor. İki yerdeki "clustered" dağılımı farklı sonuç veriyor — benchmark ile demo görsel olarak aynı ortamı temsil etmiyor. |
| 3.4 | Genel API | `CpuCollisionResult`, `CudaCollisionResult`, `CudaGridResult` üçü de pratik olarak aynı alanları taşıyor (`collision_count`, `candidate_pair_count`, `execution_time_ms`). `CudaGridResult` ek olarak `GridStats` taşıyor. Tek bir `CollisionResult` + opsiyonel `GridStats` ile birleştirilebilir. |
| 3.5 | Genel API | "Method strategy" arayüzü yok. Generic bir `BroadphaseMethod` (function pointer + ad + opsiyonel parametre yapısı) tanımlanıp benchmark döngüsü buna göre yazılırsa yeni yöntem (ör. spatial hash) eklemek tek satıra düşer. |
| 3.6 | `visualization/.../CudaSimulation.cu:148-195` | `write_vbo_kernel` her frame 6 vertex/circle yazıyor (yaklaşık `count*56` byte). Daha generic bir yaklaşım: instanced rendering veya geometry shader — fakat bu optimizasyon, hedef "modüler" olduğu için ileride çekirdek + render ayrımı kolaylaşır. |
| 3.7 | `visualization/.../CudaSimulation.cu:112-146` | Görselleştirici kendi brute-force kernel'ini taşıyor; `src/CudaBruteForce.cu` içindeki kernel'i yeniden kullanmıyor. Modüler olsa görselleştirici yalnızca integration + render katmanı yazardı. |

---

## 4. Gerçek bug'lar

| # | Yer | Sorun |
|---|---|---|
| 4.1 | `visualization/.../CudaSimulation.cu:205` | `cudaGraphicsGLRegisterBuffer(&g_vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard)` — yanlış enum. `cudaGraphicsMapFlagsWriteDiscard` `cudaGraphicsMapResources` içindir; burada `cudaGraphicsRegisterFlagsWriteDiscard` olmalı. Numerik değerleri her ikisinde de 2 olduğu için derlenip çalışıyor, ama anlam yanlış ve gelecek sürümlerde bozulabilir. |
| 4.2 | `src/CpuCollision.c:5-7,23,32` | CPU zamanı `clock()` ile ölçülüyor. `clock()` CPU **process** zamanı verir, GPU `cudaEventElapsedTime` ise wall-clock verir. Karşılaştırma adil değil, özellikle CI/sanal makinelerde çekirdek başına ölçüm farklılaşır. `clock_gettime(CLOCK_MONOTONIC, ...)` kullanılmalı; Windows için `QueryPerformanceCounter`. |
| 4.3 | `src/CudaBruteForce.cu:70-77` ve `src/CudaGrid.cu:223-253` | CUDA timing yalnızca kernel'i kapsıyor; H2D `cudaMemcpy` (circle verisi) ve D2H okuma timing dışında. Bu nedenle `speedup_vs_cpu` "kernel-only" hızlanma; başlık rapor için yanıltıcı. Ya transferleri içeren toplam süre ölçülmeli ya da CSV'ye `kernel_time_ms` ve `total_time_ms` ayrı yazılmalı. |
| 4.4 | `src/CudaGrid.cu:189-191` | Cell size constraint (`max_radius*2 ≤ cell_size`) kod içinde **doğrulanmıyor**. README bunu belirtiyor, fakat config bozulursa 8-komşu araması sessizce çarpışmaları kaçırır. Runtime check eklenmeli (`if (cell_size < 2*max_radius) return error/warn`). |
| 4.5 | `src/CudaGrid.cu:213-214` | `cudaMemset(d_cell_start, 0xFF, ...)` — `0xFFFFFFFF` int olarak `-1`. OK ama satır `4.6`'daki kontrole bağlı; `int` boyutuna ve endian'a bağlı bir varsayım. Açık `init_kernel` yazmak daha okunaklı olurdu (modülerlik). |
| 4.6 | `src/CudaGrid.cu:60-64` | `build_cell_ranges_kernel` aynı cell_id boundary tespiti için `i-1` ve `i+1` indekslerini tarıyor. Doğru yazılmış (i==0 ve i==count-1 guard'ları var) ama `__shared__` veya stream-compaction yaklaşımı kullanılarak global memory baskısı azaltılabilir — bug değil, tek "broad-phase" iddiası altında performans notu. |
| 4.7 | `visualization/.../CudaSimulation.cu:134-135` | `atomicExch(&balls[i].colliding, 1)` — değer sabit `1`, race olsa bile sonuç deterministik. Atomic gereksiz; `balls[i].colliding = 1;` aynı işi yapar ve daha hızlıdır. |
| 4.8 | `src/main.cu:5` | `int main(void)` — argv yok. CLI'dan obje sayısı / config geçilemiyor. Visualizer `argv[1]` desteği var ama benchmark binary değişken almıyor. Modüler olsa konfig dosyası veya CLI flag'leri eklenirdi. |
| 4.9 | `results/timings.csv` | Sadece header var, satır yok. Repoya boş bir çıktının check-in edilmesi tutarsız; `.gitkeep` zaten var, `timings.csv` ignore edilebilirdi. |
| 4.10 | `src/Benchmark.c:135` | `free(circles)` `run_distribution` içinde — fonksiyon `circles`'ı parametre olarak alıyor ve içeride free ediyor. Sahiplik (ownership) belirsiz; `Benchmark.c:190,196` çağrısının `free`'lemediği malloc'u alt fonksiyon free'liyor. Hata yolundaki `return 0` öncesi free yok → leak (allocation başarısızsa zaten çağrı içeriden yapılmamış olur — bu özel durum OK ama API kontratı net değil). |
| 4.11 | `visualization/.../main.cpp:115` | `Color{8, 10, 14, 255}` — C++. C için `(Color){8, 10, 14, 255}` compound literal olmalı (main.c'ye dönüşle birlikte). |

---

## 5. Tutarsızlıklar

| # | Konu | Detay |
|---|---|---|
| 5.1 | "Pure C" iddiası vs `.cpp` ve C++ kullanımı | `README.md:5` ile çelişen `main.cpp`. |
| 5.2 | Visualization README "uniform-grid mode planned" diyor | Şu an sadece brute-force çalışıyor; visualizer asıl projenin ana katkısı olan **broad-phase grid'i göstermiyor** — bu ironi: "broad-phase visualizer" olmalıydı. |
| 5.3 | Cluster üretimi farkı | Madde 3.3 — benchmark ve görselleştirici aynı "clustered" dağılımı üretmiyor. |
| 5.4 | RNG seed davranışı | `cuda_visualizer_reset` her zaman sabit seed `202405u` kullanıyor (`CudaSimulation.cu:246`). Reset hep aynı manzarayı veriyor. Demo için fena değil ama beklenmedik. |
| 5.5 | `Benchmark.h` `MAX_BENCHMARK_OBJECT_COUNTS` = 5, `MAX_GRID_CELL_SIZES` = 4 | Sabit dizi boyutu — generic değil. Konfig daha esnek olabilirdi (heap allocate veya dinamik). |
| 5.6 | Header guard tutarsızlığı | Bütün `.h` ve `.cuh`'lerde `#pragma once` var, OK. Ama `extern "C"` blokları `Circle.h`, `CpuCollision.h`, `CudaBruteForce.cuh`, `CudaGrid.cuh`, `DataGenerator.h`, `Benchmark.h`, `RaylibInteropTypes.cuh`'de var; `CudaUtils.cuh`'de yok (yalnız `CUDA_CHECK` makrosu içerir, ama yine de tutarlılık için eklenebilir). |
| 5.7 | CSV sütunu `grid_cell_size` CPU/CUDA brute satırlarında 0.00 | `Benchmark.c:42` `memset(0)` ile başlatılıyor, sorun değil ama "kullanılmıyor" sütun değeri açıkça `NA` veya boş bırakılabilirdi (raporlama netliği). |
| 5.8 | `visualization/.../CMakeLists.txt:32` | `target_include_directories` `../../include` ekliyor (ana proje header'ları için), iyi; ama görselleştirici `Circle.h` veya benchmark başlıklarını gerçek anlamda kullanmıyor — sadece `CudaUtils.cuh` için. Ortak modül kullanımı tasarlansa daha tutarlı olur. |

---

## 6. Önerilen yapısal yön (uygulama planı, onaylanırsa)

Sıralı, küçük PR'larla:

### Adım A — Saf C dönüşümü (görselleştirici)
- `visualization/raylib-cuda-interop/src/main.cpp` → `main.c`.
- `<cstdio>` vb. yerine `<stdio.h>`, `std::` kalıntılarını temizle.
- `CMakeLists.txt`: `LANGUAGES C CUDA`, target dosya listesi güncelle.
- `Color{...}` → `(Color){...}`.
- `<windows.h>` ve `<GL/gl.h>` doğrudan include'larını kaldır; raylib + rlgl
  zaten taşınabilir. Eğer `cudaGraphicsGLRegisterBuffer` için bir GL header
  gerekiyorsa, raylib'in kendi mekanizmasıyla VBO id'si zaten var.

### Adım B — Ortak modüller (modüler/generic)
- Yeni header `include/CollisionMath.h` → `__device__ __host__ inline int circles_overlap(...)`.
  `CpuCollision.c`, `CudaBruteForce.cu`, `CudaGrid.cu`, ve `CudaSimulation.cu` bu header'ı kullansın.
- Yeni header `include/Rng.h` → ortak LCG (`__device__ __host__`).
  `DataGenerator.c` ve `CudaSimulation.cu` bunu kullansın.
- Yeni header `include/CollisionResult.h` → tek `CollisionResult` struct'ı +
  opsiyonel `GridStats` field'ı. Üç ayrı tipi kaldır.

### Adım C — Bug fix'leri
- `CudaSimulation.cu:205` → `cudaGraphicsRegisterFlagsWriteDiscard`.
- `CpuCollision.c` → `clock_gettime(CLOCK_MONOTONIC)`-tabanlı `now_ms()` helper.
- `CudaGrid.cu:run_cuda_uniform_grid` başına: `if (cell_size < 2.0f * max_radius_seen) {...}` runtime check ve uyarı.
- CUDA brute/grid timing'ine memcpy'ları dahil eden veya CSV'ye ek bir
  `kernel_time_ms` kolonu eklemek (kullanıcı kararı).
- `CudaSimulation.cu` → `atomicExch` yerine düz atama.

### Adım D — Visualizer parite
- `visualization`'daki cluster init kernel'i kaldırılıp, ana proje
  `generate_clustered_circles` benzer mantığı (rastgele cluster merkezleri) ile
  değiştirilsin. CPU'da host-side init + tek seferlik H2D copy yapılabilir;
  CPU mirroring değil, sadece başlangıç data transferi.
- "CUDA grid mode" visualizer'a eklensin (README'nin söz verdiği özellik).

### Adım E — Generic broadphase API
- `include/Broadphase.h`:
  ```c
  typedef CollisionResult (*BroadphaseFn)(const Circle*, size_t, const void* params);
  typedef struct BroadphaseMethod {
      const char* name;
      BroadphaseFn run;
      const void* params;
  } BroadphaseMethod;
  ```
- `Benchmark.c` bir `BroadphaseMethod[]` üzerinde dolaşsın, yeni yöntem
  eklemek tek satıra düşer.

---

## 7. Kritik dosyalar (referans)

- `src/main.cu` — entry, .c'ye taşınacak.
- `src/CpuCollision.c` — timing fix.
- `src/CudaBruteForce.cu` — kernel ortak module'a referans.
- `src/CudaGrid.cu` — runtime cell-size check, kernel ortak module'a.
- `src/Benchmark.c` — generic method tablosu, ownership netliği.
- `src/DataGenerator.c` — RNG modülüne extract.
- `include/Benchmark.h` — sabit dizi yerine esnek konfig.
- `visualization/raylib-cuda-interop/src/main.cpp` → `.c`.
- `visualization/raylib-cuda-interop/src/CudaSimulation.cu` — register flag fix,
  cluster init parite, atomic temizlik, ortak header kullanımı.
- `visualization/raylib-cuda-interop/CMakeLists.txt` — `LANGUAGES C CUDA`.

---

## 8. Doğrulama

- Build:
  ```bash
  mkdir build && cd build
  cmake ..
  cmake --build . -j
  ./collision_benchmark
  ```
  CSV satırları `results/timings.csv`'ye yazılmalı; CPU collision count ile
  CUDA brute force collision count birbirine eşit veya 0 farklı olmalı
  (float belirsizliği).

- Görselleştirici (Windows/NVIDIA):
  ```powershell
  cd visualization/raylib-cuda-interop/build
  cmake --build . --config Release
  .\Release\raylib_cuda_visualizer.exe 2500
  ```
  Pencere açılmalı, 60 FPS civarı, mavi/kırmızı particle'lar, metrics ms olarak
  gösterilmeli.

- Birim doğrulama (öneri): küçük bir test C dosyası (`tests/test_collision.c`)
  basit bilinen kümelerle (örn. 3 daire, 1 çift çakışıyor) CPU ve CUDA brute
  sonuçlarının eşitliğini iddia etsin.

---

# Literatür İncelemesi ve Geliştirme Yol Haritası (Mayıs 2026 araması)

## Context

Proje şu an bir CPU brute force baseline + CUDA brute force + CUDA uniform
grid (Thrust `sort_by_key` ile) içeriyor — bu literatürde **2007-2010
döneminin standart "GPU particle/uniform-grid"** mimarisi. Yöntem sağlam ama
2026'da hem akademik hem pratik açıdan başka 5-6 alternatif daha var. Aşağıda
şunları sıralıyorum: (1) konunun temel referansları, (2) bu projenin
literatürdeki konumu, (3) faz faz geliştirme yol haritası, (4) daha iyi
paralelleştirme teknikleri, (5) akademik/teknik sunum için yapılması gereken
işler.

## 1. Temel literatür (kategorize)

### A. Bu projeyle **birebir** örtüşen klasikler (referans gibi davran)

- **Le Grand, S. — "Broad-Phase Collision Detection with CUDA" — GPU Gems 3, Chapter 32 (NVIDIA, 2007)**
  ([sayfa](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda)).
  Uniform grid + cell-id'ye göre radix sort + cell_start/cell_end yaklaşımının
  ilk kanonik tarifi. Bu projenin `CudaGrid.cu` dosyası tam olarak bu metodun
  bir varyantı. Raporda zorunlu atıf.

- **Green, S. — "Particle Simulation using CUDA" — NVIDIA Whitepaper (2007/2010/2012)**
  ([2010 PDF](https://developer.download.nvidia.com/assets/cuda/files/particles.pdf),
  [2012 PDF](https://developer.download.nvidia.com/compute/DevZone/C/html_x64/5_Simulations/particles/doc/particles.pdf)).
  CUDA SDK'daki `particles` örneğinin teknik raporu. Atomic-counter ve
  radix-sort tabanlı iki uniform grid kuruluş yöntemi karşılaştırıyor; "cell
  size = 2 × radius" tasarım kuralı buradan geliyor — README'de zaten geçen
  kısıt.

### B. Daha gelişmiş klasikler (yeni metod eklemek için)

- **Karras, T. — "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees" — HPG 2012**
  ([NVIDIA PDF](https://research.nvidia.com/sites/default/files/pubs/2012-06_Maximizing-Parallelism-in/karras2012hpg_paper.pdf)).
  Modern GPU LBVH (Linear BVH) yapısının temeli. Morton kod → radix sort →
  binary radix tree → AABB propagation. Bu projeye **uniform grid'den sonra
  eklenecek doğal ikinci yöntem**.
  - Sadeleştirilmiş anlatım: NVIDIA blog "Thinking Parallel" üçleme:
    [Part I](https://developer.nvidia.com/blog/thinking-parallel-part-i-collision-detection-gpu/),
    [Part II](https://developer.nvidia.com/blog/thinking-parallel-part-ii-tree-traversal-gpu/),
    [Part III](https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/).
  - Açık kaynak referans implementasyon: [ToruNiina/lbvh](https://github.com/ToruNiina/lbvh),
    [jerry060599/KittenGpuLBVH](https://github.com/jerry060599/KittenGpuLBVH),
    [Bullet3 b3GpuParallelLinearBvh](https://github.com/bulletphysics/bullet3/blob/master/src/Bullet3OpenCL/BroadphaseCollision/b3GpuParallelLinearBvh.h).

- **Liu, F. et al. — "Real-time Collision Culling of a Million Bodies on Graphics Processing Units" — SIGGRAPH Asia 2010**
  ([PDF](https://graphics.ewha.ac.kr/gSaP/gSaP.pdf),
  [GitHub](https://github.com/liufububai/GPU-Sweep-Prune-Collision-Detection)).
  GPU üzerinde sweep-and-prune (sort-and-sweep). Frame-to-frame koherans
  kullanıyor, dinamik sahnelerde uniform grid'den iyi olabiliyor.

- **Tang, M. et al. — "PSCC: Parallel Self-Collision Culling with Spatial Hashing on GPUs"**
  ([PDF](https://min-tang.github.io/home/PSCC/files/pscc.pdf)).
  Açık spatial hash table (kapalı sınırlı uniform grid yerine) — sahne
  sınırları sabit olmadığında kullanılır.

- **Teschner, M. et al. — "Optimized Spatial Hashing for Collision Detection of Deformable Objects"**
  ([PDF](https://matthias-research.github.io/pages/publications/tetraederCollision.pdf)).
  Spatial hash'in en çok atıf alan tarifi.

### C. Modern (2023-2026)

- **Sui, S. et al. — "Hardware-Accelerated Ray Tracing for Discrete and Continuous Collision Detection on GPUs" — ICRA 2025**
  ([arXiv 2409.09918](https://arxiv.org/abs/2409.09918),
  [proje sayfası](https://ssz990220.github.io/publication/RTCD)).
  RTX RT-core'ları broadphase kovalamak için kullanıyor; geleneksel kernel
  tabanlı yöntemlere göre belirli iş yüklerinde 3× hızlanma.

- **"Mochi: Collision Detection for Spherical Particles using GPU Ray Tracing"**
  ([arXiv 2402.14801](https://arxiv.org/abs/2402.14801)).
  Bu projeyle **birebir aynı problem** (küresel parçacıklar) ama RT cores
  üzerinde. Uniform grid ve hash map baseline'larına karşı kıyaslıyor — bu
  proje ileride bunlara karşı kıyaslanacak referans olabilir.

- **"Rethinking Collision Detection on GPU Ray Tracing Architecture"**
  ([arXiv 2604.23520](https://arxiv.org/html/2604.23520v1)).

### D. Yardımcı / arka plan

- **Karras, T. & Aila, T. — "Fast Parallel Construction of High-Quality BVHs" — HPG 2013**
  ([PDF](https://research.nvidia.com/sites/default/files/pubs/2013-07_Fast-Parallel-Construction/karras2013hpg_paper.pdf)).
  LBVH'ı SAH-iyileştirme ile güçlendiren takip çalışma.

- **Wang, B. et al. — "Efficient BVH-based Collision Detection Scheme with Ordering and Restructuring"**
  ([PDF](http://gamma.cs.unc.edu/PAPERS/WangEurographics2018.pdf)).
  Eurographics 2018 — BVH'ı dinamik sahnelerde refit + restructure ile
  kullanma stratejisi.

- **Lefebvre, S. & Hoppe, H. — "Perfect Spatial Hashing"**
  ([PDF](https://hhoppe.com/perfecthash.pdf)).
  Statik sahnelerde collision-free hash; demo için fazla iş ama referans
  olarak iyi.

- **Nocentino, A. — "Optimizing Memory Access on GPUs using Morton Order Indexing"**
  ([PDF](https://www.nocentino.com/Nocentino10.pdf)).
  Morton/Z-order index'in coalescing'e somut etkisi — Faz A.1'in
  motivasyonu.

## 2. Mevcut implementasyonun literatürdeki konumu

| Boyut | Bu proje | Literatür durumu |
|---|---|---|
| Broadphase yöntemi | Uniform grid (cell_id sort + 9-komşu) | 2007 standart; tek başına yetersiz değil ama tek seçenek olarak modern değil |
| Sort | Thrust `sort_by_key` | OK, ama [CUB](https://developer.nvidia.com/blog/thrust-cub-1-11/) doğrudan kullanılırsa temp storage yeniden kullanılabilir; tekrarlı çalıştırmada %5-15 hızlanma yaygın |
| Cell hash | Row-major (`cell_y*W + cell_x`) | [Morton/Z-order](https://en.wikipedia.org/wiki/Z-order_curve) memory locality için daha iyi; Green 2010 önerisi |
| Hücre boyutu | Tek sabit `cell_size` | Sahnede objeler farklı boyuttaysa hierarchical grid (Lefebvre-style) lazım; bu projede tek radius aralığı olduğu için kritik değil |
| AABB / dar faz | Yok (sadece daire-daire) | Genelleştirme için AABB-aware bir broadphase + narrow phase ayrımı standart; daireden çıkıp poligonlara geçmek istenirse şart |
| Distribusyonlar | Uniform + clustered | Yeterli; literatürde grid + Gaussian + lattice + scaling-test yaygın |
| Doğrulama | Yok | Tüm modern referans implementasyonlar (Bullet, ToruNiina/lbvh) parity test'iyle kıyaslar |
| Profiling | Yok | Nsight Compute / `nvprof` çıktısı modern GPU çalışmalarında zorunlu |
| Karşılaştırma noktaları | Sadece kendi metodları | Adil rapor için en az bir 3rd-party (NVIDIA particles SDK örneği) ile birebir karşılaştırma standart |

**Özet:** Bu proje "uniform grid baseline" düzeyinde sağlam; literatür tarafıyla
örtüşüyor ama "modern" ya da "yeni katkı" iddiasında değil. Sunum açısından
strateji şu olmalı: ya **mükemmel bir uniform grid + analiz çalışması** olarak
sun (mevcut hâli buna yakın), ya da **bir veya iki ek yöntem ekleyip
karşılaştırma** sun (Faz B).

## 3. Geliştirme yol haritası (faz/effort sıralı)

### Faz A — Tek PR kazanımları (1-2 hafta, çok yüksek değer)

| # | Madde | Effort | Etki |
|---|---|---|---|
| A.1 | **Morton/Z-order cell ID** — `compute_cell_keys_kernel` içinde `cell_y*W+cell_x` yerine 2D Morton kodu kullan. `__device__ static unsigned int morton2d(unsigned int x, unsigned int y)` — 16-bit interleave. Kıyasla: clustered'da %10-30 daha az candidate pair tarama süresi yaygın (Nocentino). | S | M-H |
| A.2 | **Doğrudan CUB radix sort** — `thrust::sort_by_key` yerine `cub::DeviceRadixSort::SortPairs` ile temp storage'ı bir kez allocate edip yeniden kullan. Frame-by-frame senaryoda (visualizer) gerçek kazanç. | S | M |
| A.3 | **Multi-stream pipeline** — H2D, kernel ve D2H'ı 2-3 stream'e böl; brute-force yapısında transferleri kerneller arasına gizle. CUDA brute force'ta %20-40 toplam süre tasarrufu olur (memcpy bandwidth-bound iken). | S-M | M |
| A.4 | **Parity testi** (`tests/test_parity.c`) — küçük (100), orta (10k) ve patolojik (clustered, dense) durumlarda CPU ≡ CUDA brute ≡ CUDA grid `collision_count` doğrulaması. Float drift için `±max(1, 0.001×count)` toleransı. | S | H (rapor güvenilirliği) |
| A.5 | **Nsight çıktısı CSV ekstraksiyonu** — `ncu --csv --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed` çağırıp `results/profile.csv` üret. Roofline modelinde her yöntemin yeri görünsün. | M | H (sunum için zorunlu) |
| A.6 | **Plot betiği** (`scripts/plot_results.py`) — matplotlib ile 4 grafik: (i) total_time vs N (her yöntem ayrı eğri), (ii) speedup vs N, (iii) candidate_pair_count vs N, (iv) clustered vs uniform için max_objects_in_cell histogramı. | S | H |

### Faz B — Yeni broadphase yöntemleri (3-6 hafta, orta-yüksek değer)

Hedef: 3 yöntem yerine 5-6 yöntemli karşılaştırma. Her birinin `BroadphaseFn`
arayüzüne uyduğunu unutma — `Benchmark.c::build_methods`'a tek satır eklemek
yeterli olur.

| # | Yöntem | Atıf | Effort |
|---|---|---|---|
| B.1 | **Spatial hash (open hash table)** — sahne sınırından bağımsız, ortalama O(1) lookup. `hash(cell_x, cell_y) = (cell_x * P1) XOR (cell_y * P2) mod tableSize`. Hash collision'ları zincir/probe ile çöz. | Teschner 2003, Tang PSCC | M |
| B.2 | **Sweep-and-Prune (sort-and-sweep)** — AABB'leri tek eksende sırala, scan ile aktif liste tut. GPU'da Liu 2010 yapısıyla. Frame-to-frame koherans olduğunda dinamik sahnelerde grid'i yenebilir. | Liu 2010 | M-L |
| B.3 | **LBVH (Linear BVH)** — Karras 2012. Morton kod → radix sort → binary radix tree → AABB propagation → traversal. Geniş uygulanabilir; akademik raporda **belirleyici** yöntem. | Karras 2012 | L |
| B.4 | **Hierarchical grid** — birden fazla cell_size'lı grid katmanı. Bu projedeki radius aralığı dar olduğu için zenginlik kazancı sınırlı, ama "max_radius'u 50× yapıp deneyin" senaryosunda fark gözle görülür. | Eitz & Lixu 2007 | M |
| B.5 | **GPU brute force tiled (shared memory)** — özellikle N ≤ 16k için: her blok TILE_SIZE objeyi shared memory'ye yükler, başka blokların objeleriyle kıyaslar. Mevcut brute force kerneli zaten %30-50 hızlanır. Klasik nbody pattern. | NVIDIA SDK nbody | S |

### Faz C — Modern donanım (opsiyonel, donanıma bağlı)

| # | Madde | Şart |
|---|---|---|
| C.1 | **OptiX/RT-cores tabanlı broadphase** (Mochi, Sui ICRA 2025 stili) — daireleri AABB primitive'leri olarak BVH'a yerleştir, ray-AABB intersection'ı broadphase olarak kullan | RTX 2000+, OptiX 7+ |
| C.2 | **Tensor core kullanımı** — broadphase'de mantıklı değil; bu maddeyi rapora "denenmeyen alternatif" olarak yaz | — |
| C.3 | **Multi-GPU** — `cudaMemcpyPeer` ile sahneyi parçala, her GPU kendi bölümünün collision'ını sayar | 2+ GPU |

### Faz D — Sunum / metodoloji polajı (1 hafta)

| # | Madde |
|---|---|
| D.1 | **Akademik rapor şablonu** (`docs/report.md` veya `report.tex`) — Abstract / Background / Methods / Implementation / Experiments / Discussion / Limitations / Future Work / References. Yukarıdaki literatür kaynaklarını BibTeX'e çevir |
| D.2 | **Kıyaslama matrisi** — tüm yöntemler × {1k, 10k, 100k, 1M} × {uniform, clustered, gaussian} ızgarası. CSV → tablo otomasyonu |
| D.3 | **Roofline grafiği** — A.5'deki metric'lerden Nsight tarzı bandwidth-vs-compute scatter |
| D.4 | **Animasyonlu visualizer demo** — `R` ile reset, `G` ile grid moduna geçiş, candidate pair sampling overlay'i (10 random pair'i çiz) |
| D.5 | **Sunum slide'ları** (`docs/slides/`) — Quarto/Marp ile Markdown → HTML. Her yöntem için: anim diagramı + kernel görseli + metrics tablosu |
| D.6 | **README'ye "Comparison with literature" bölümü** — bu proje vs Le Grand / Green / NVIDIA particles SDK kaba kıyaslama tablosu |

## 4. Daha iyi paralelleştirme teknikleri (mevcut yöntemlere uygulanabilir)

Bunlar yeni broadphase metodu değil — **var olan kerneller için optimizasyon**.

### 4.1 Memory hijyeni

- **Coalescing**: `Circle` struct'ı şu an AoS (Array of Structs). `x[]`, `y[]`, `radius[]` ayrı array'lerine (SoA) bölünürse warp başına 32 float ardışık erişim alır → bandwidth %2-3× iyileşir. CUDA programming guide'ın ilk önerisi.
- **Z-order data layout**: A.1 madddesi (cell_id Morton). Ek olarak: sıralama sonrası objeleri yeniden permutate ederek **fiziksel olarak Z-order'a göre dizilmiş `Circle`** elde etmek; cache locality artar (Green 2010, Nocentino 2010).
- **Texture/`__ldg__`**: read-only objeler için `__ldg(&circles[i])` cache'i ısıtır; yeni nesil GPU'larda compiler zaten yapıyor ama eski mimarilerde belirgin.

### 4.2 Block-level kooperasyon

- **Shared memory tile** (B.5'in fikri brute force için, ama grid için de uygulanabilir): bir blok bir komşu hücre topluluğunu shared'e yükler, blok içi tüm thread'ler üzerinden tarama. Global memory traffic %10-30 düşer.
- **Cooperative groups (`coalesced_group`, `tile_partition<32>`)**: warp-wide reduction ile `local_collisions` ve `local_candidates` toplamlarını `__shfl_sync` üzerinden topla, sonra block leader bir kez `atomicAdd` çağırsın → atomic baskısı 32× azalır.

### 4.3 Atomic baskısını azaltma

Şu an her thread `local_candidates`'ı blok düzeyinde toplamadan doğrudan
global `atomicAdd` yapıyor. Plan:
1. Warp-level `__shfl_xor_sync` ile 32 thread'in toplamını birleştir.
2. Blok-level: lane 0'lar shared memory'deki bir slot'a yazsın.
3. Block leader (thread 0) shared toplamı tek `atomicAdd` ile global'e ekler.

Bu değişiklik özellikle 100k+ N için kerneli %15-25 hızlandırır.

### 4.4 Stream paralelliği

CUDA brute force'ta H2D copy + kernel + D2H sıralı. Bunu 4 stream'e böl:
- Stream 0: H2D (tüm objeler tek seferde)
- Stream 1-3: 3 ayrı tile üzerinde kernel
- Stream 0: D2H (sayaç değerleri)

Kernel ve memcpy'yi overlap ettiği için toplam süre `max(H2D, kernel)`'e
yaklaşır. Shorter kernel'lerde kazanç %30-50.

### 4.5 Kernel füzyonu

`run_cuda_uniform_grid` şu an 4 ayrı kernel + 1 thrust sort çağırıyor:
init_cell_ranges → compute_cell_keys → sort → build_cell_ranges → grid_collision.
- `compute_cell_keys` ile `init_cell_ranges` aynı launch'a sığabilir
  (independent).
- Çok küçük N'de kernel launch overhead'i baskın; 1k objeden küçük
  benchmark'larda füzyon önemli.

### 4.6 Persistent threads

Modern GPU'larda iş yüküne göre dinamik denge. Bu projede iş yükü statik
olduğu için fayda sınırlı; LBVH eklenirse traversal'da değer kazanır.

## 5. Sunum açısından "nasıl daha iyi anlatırız?"

### 5.1 Hikaye yapısı

> "Brute-force GPU bile CPU'yu yenemediği patolojik N değerleri var (cache + transfer overhead). Broadphase devreye girdiğinde bu eşik hangi N'e iner? Hangi distribusyonlar broadphase'in faydasını yok eder? Bu projede ölçtük."

Bu cümle hem giriş hem ana bulguya çıkış sağlar. Slide 1, 2 ve son grafik bu
cümlenin etrafında dönsün.

### 5.2 Görsel argüman

- **Performance plot**: log-log scale, x = N, y = ms. CPU baseline + 2-3 CUDA eğrisi. Crossing point'lerini işaretle.
- **Candidate pair plot**: x = N, y = candidate pair count. Brute force = N(N-1)/2 düz çizgisi referans, grid eğrisi altta çok daha düşük.
- **Distribution sensitivity**: clustered'a geçince candidate count'un nasıl patladığını gösteren bar chart.
- **Roofline**: D.3 — yöntemler arithmetic intensity vs achieved bandwidth.

### 5.3 Validasyon argümanı

> "3 farklı yöntem aynı sayıyı veriyor → algoritmik doğruluk garantisi. Sayıyı veren parity testi reposunda, CI'da çalışıyor."

Bu olmadan akademik sunum eksik kalır.

### 5.4 Limitations bölümü dürüst yazılmalı

- Sadece dairesel objeler (AABB-poligon broadphase yok)
- Dar faz (narrow phase) yok — gerçek motorda gerekli
- Tek GPU
- Cell-size uniform; hierarchical denenmedi (Faz B'de)
- `cell_size < 2 * max_radius` koruması var ama dinamik cell_size adaptasyonu yok
- RT-core tabanlı modern alternatif (Mochi, ICRA 2025) denenmedi (Faz C)

### 5.5 Reference implementations'a karşı kıyaslama

NVIDIA `cuda-samples/Simulations/particles/` aynı problemi çözüyor (Green
2010). 10k-100k N için bu projenin grid yöntemi vs NVIDIA SDK'nın particle
örneği yan yana çalıştırılıp ms karşılaştırılırsa rapor "third-party'den
%X uzakta / yakın" diyebilir. Sunum'a kalıcı ağırlık katar.

## 6. Önerilen aksiyon önceliği

Eğer bir tek sıra istersen:

1. **A.4 parity testi** — bu olmadan diğerlerini optimize ederken sessizce kırarsın
2. **A.6 plot betiği + A.5 Nsight CSV** — ölçemediğin şeyi iyileştiremezsin
3. **A.1 Morton cell-id** — small change, immediate sunum noktası ("Z-order coalescing eklendiğinde %X")
4. **B.3 LBVH** — sunumu "uniform grid baseline + modern alternative" seviyesine taşır
5. **B.5 tiled brute force** — kolay kazanç
6. **D.1 rapor şablonu** — yazmaya hızı kaybetmeden başla
7. **A.3 multi-stream** — total_time CSV kolonunda gözle görülür kazanç
8. **C.1 OptiX** — sadece "future work" başlığı altında ama 2025'te ilgili olduğu için en azından README'de adı geçsin

## 7. Doğrulama (yeni geliştirme bittiğinde nasıl bilinir?)

- Tüm metodlar, tüm distribusyonlarda **aynı `collision_count`'u** üretmeli
  (parity testi pass).
- `total_time_ms` ve `kernel_time_ms` arasındaki fark belirgin (memcpy'nin
  payı görünür).
- Plot betiği 4 grafiği üretebilmeli, her bar/line için en az 5 farklı N.
- Nsight Compute çıktısında `dram__throughput` ≥ %50 (memory-bound olmalı,
  brute force hariç) — değilse memory access pattern bozuk demektir.
- Visualizer 60 FPS @ 5k-10k obje, grid modunda 25k objeye ölçeklensin.
- LBVH eklenirse: aynı parity test'i geçmeli, total_time clustered'da grid'i
  yenmeli.

## Kaynaklar (özet)

Klasik:
- [GPU Gems 3 Ch.32 — Le Grand 2007](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda)
- [Particle Simulation using CUDA — Green 2010](https://developer.download.nvidia.com/assets/cuda/files/particles.pdf)
- [Optimizing Memory Access on GPUs — Nocentino 2010](https://www.nocentino.com/Nocentino10.pdf)
- [Optimized Spatial Hashing — Teschner 2003](https://matthias-research.github.io/pages/publications/tetraederCollision.pdf)

LBVH ve sweep-and-prune:
- [Karras HPG 2012](https://research.nvidia.com/sites/default/files/pubs/2012-06_Maximizing-Parallelism-in/karras2012hpg_paper.pdf)
- [Karras & Aila HPG 2013](https://research.nvidia.com/sites/default/files/pubs/2013-07_Fast-Parallel-Construction/karras2013hpg_paper.pdf)
- [Liu SIGGRAPH Asia 2010 (gSaP)](https://graphics.ewha.ac.kr/gSaP/gSaP.pdf)
- [Wang Eurographics 2018 — BVH ordering](http://gamma.cs.unc.edu/PAPERS/WangEurographics2018.pdf)
- [PSCC — Tang](https://min-tang.github.io/home/PSCC/files/pscc.pdf)

Modern (2024-2026):
- [Mochi: GPU Ray Tracing CD — arXiv 2402.14801](https://arxiv.org/abs/2402.14801)
- [Hardware-Accelerated RT for CD — arXiv 2409.09918](https://arxiv.org/abs/2409.09918)
- [Rethinking CD on RT Architecture — arXiv 2604.23520](https://arxiv.org/html/2604.23520v1)

Açık kaynak referans implementasyonlar:
- [ToruNiina/lbvh](https://github.com/ToruNiina/lbvh)
- [jerry060599/KittenGpuLBVH](https://github.com/jerry060599/KittenGpuLBVH)
- [Bullet3 b3GpuParallelLinearBvh](https://github.com/bulletphysics/bullet3/blob/master/src/Bullet3OpenCL/BroadphaseCollision/b3GpuParallelLinearBvh.h)
- [liufububai/GPU-Sweep-Prune-Collision-Detection](https://github.com/liufububai/GPU-Sweep-Prune-Collision-Detection)
- [NVIDIA cuda-samples particles](https://github.com/NVIDIA/cuda-samples) (Simon Green'in örneği)

Tooling:
- [NVIDIA Thrust + CUB 1.11 update](https://developer.nvidia.com/blog/thrust-cub-1-11/)
- ["Thinking Parallel" üçlemesi — NVIDIA blog](https://developer.nvidia.com/blog/thinking-parallel-part-i-collision-detection-gpu/)
- [Maximizing Parallel Hash Maps on GPUs — NVIDIA blog](https://developer.nvidia.com/blog/maximizing-performance-with-massively-parallel-hash-maps-on-gpus/)
