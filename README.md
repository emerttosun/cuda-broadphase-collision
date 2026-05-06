# CUDA Broad-Phase Collision Detection

CPU baseline, CUDA brute force ve CUDA uniform-grid yöntemleri için 2B daire çarpışma benchmark'ı + canlı raylib görselleştiricisi. Tek bir CMake çağrısıyla iki binary üretir:

- `collision_benchmark` — CSV çıkaran benchmark (`results/timings.csv`).
- `raylib_cuda_visualizer` — CUDA-OpenGL interop ile canlı simülasyon.

## Gereksinimler

- **Visual Studio 2022** + "Desktop development with C++" workload.
- **CMake ≥ 3.24** (`winget install Kitware.CMake`).
- **CUDA Toolkit 12.x** — https://developer.nvidia.com/cuda-toolkit-archive
  (GTX 10-serisi / Pascal için CUDA 12.x şart, CUDA 13+ Turing ve üstünü ister.)
- **raylib kaynağı** `external/raylib` altında. Yoksa proje kökünden:
  ```powershell
  git clone https://github.com/raysan5/raylib.git external/raylib
  ```

vcpkg gerekmiyor — raylib projeyle birlikte kaynaktan derlenip statik linkleniyor.

## 1. Build öncesi: `CMakePresets.json`'u kendi makinene uyarla

`CMakePresets.json` dosyasını aç. `default` preset'i şu şekilde:

```json
{
    "name": "default",
    "generator": "Visual Studio 17 2022",
    "architecture": "x64",
    "binaryDir": "${sourceDir}/build",
    "toolset": "cuda=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1",
    "cacheVariables": {
        "CMAKE_CUDA_COMPILER": "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/bin/nvcc.exe",
        "BUILD_VISUALIZER": "ON"
    }
}
```

Düzeltmen gerekenler:

### a) CUDA sürümü

`v12.1` iki yerde geçiyor — kendi yüklü sürümünle değiştir, ikisi de aynı olmalı. Örneğin CUDA 12.6 yüklüyse `v12.1` → `v12.6`.

### b) Visual Studio sürümü

VS 2022 yoksa `"generator"` satırını değiştir:
- VS 2019 için: `"Visual Studio 16 2019"`

### c) GPU mimarisi (opsiyonel)

Varsayılan değer `native` — yerel GPU'yu otomatik algılar. Sabit bir mimariye derlemek istersen `cacheVariables` içine `CMP674_CUDA_ARCHITECTURES` ekle. Aşağıdaki örnek RTX 30xx için:

```json
"cacheVariables": {
    "CMAKE_CUDA_COMPILER": "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/bin/nvcc.exe",
    "BUILD_VISUALIZER": "ON",
    "CMP674_CUDA_ARCHITECTURES": "86"
}
```

GPU'na göre değer:

| GPU                   | Değer |
| --------------------- | ----- |
| GTX 10xx (Pascal)     | `61`  |
| GTX 16xx / RTX 20xx   | `75`  |
| RTX 30xx              | `86`  |
| RTX 40xx              | `89`  |

Birden fazla mimari için tırnak içinde noktalı virgülle ayır: `"75;86;89"`.

## 2. Build & çalıştır

Proje kökünden (`CMP674/`):

```powershell
cmake --preset default
cmake --build --preset default
```

Çıktılar:

```text
build\Release\collision_benchmark.exe
build\visualization\raylib-cuda-interop\Release\raylib_cuda_visualizer.exe
```

Çalıştır:

```powershell
.\build\Release\collision_benchmark.exe
.\build\visualization\raylib-cuda-interop\Release\raylib_cuda_visualizer.exe 2500
```

`2500` görselleştiricinin simüle edeceği daire sayısı. Benchmark `results/timings.csv` dosyasını çalıştırıldığı dizine yazar.

Preset'i değiştirip yeniden build alacaksan önce `build/` klasörünü sil:

```powershell
Remove-Item -Recurse -Force build
```

### Sadece benchmark (raylib istemiyorsan)

```powershell
cmake --preset benchmark-only
cmake --build --preset benchmark-only
```

## Görselleştirici kontrolleri

- `Space` — pause / resume
- `C` — uniform / clustered dağılım
- `G` — mod değiştir (CUDA brute force → CUDA uniform grid → CPU brute force)
- `+` / `-` — obje sayısını ±500 değiştir (sınırlar: 100..20000)
- `R` — reset
- `Esc` — çıkış

CPU brute force modu O(N²) çalıştığı için yüksek N'de FPS belirgin şekilde düşer — overlay'deki FPS ve "CPU compute" satırı CUDA'ya göre doğal bir karşılaştırma sağlar.
