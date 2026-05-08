# Raylib + CUDA-OpenGL Interop Visualizer

Canlı CUDA görselleştiricisi: raylib pencereyi/girdiyi/UI'ı yönetir, CUDA simülasyon + çarpışma hesabı yapar ve OpenGL VBO'sunu `cudaGraphicsGLRegisterBuffer` ile mapleyerek vertex'leri doğrudan yazar. Her daire iki üçgen olarak çizilir; köşeleri fragment shader maskeler.

## Build

Tercih edilen yol: kök dizinden tek seferde derle (raylib `external/raylib` altından otomatik bulunur ve subdirectory olarak derlenir, vcpkg gerekmez):

```powershell
cd <repo root>
cmake --preset default
cmake --build --preset default
.\build\visualization\raylib-cuda-interop\Release\raylib_cuda_visualizer.exe 2500
```

Detaylı talimatlar (CMakePresets'i kendi makinene uyarlama, CUDA sürümü, GPU mimarisi vb.) için kök `README.md`'ye bak.

### Standalone

Geriye dönük uyumluluk için bu dizinden de derlenebilir; bu modda kök CMake projesi `cmp674_core`'u üretmek için tekrar dahil edilir:

```powershell
cd visualization/raylib-cuda-interop
cmake -S . -B build
cmake --build build --config Release
.\build\Release\raylib_cuda_visualizer.exe 2500
```

## Kontroller

- `Space` — pause / resume
- `C` — uniform / clustered dağılım
- `V` — narrow / mixed / extreme radius profili
- `G` — mod değiştir: CUDA brute force → CUDA uniform grid → CPU brute force → CUDA LBVH
- `+` / `-` — obje sayısını ±500 değiştir (100..20000)
- `R` — mevcut dağılımı sıfırla
- `Esc` — çıkış

## Render

CUDA-OpenGL interop ile particle verisi CPU'ya hiç kopyalanmaz; CUDA üçgen vertex'lerini doğrudan rlgl-managed VBO'ya yazar.

- Mavi: çarpışmıyor
- Kırmızı: çarpışıyor
- Sol üstte canlı metrikler
