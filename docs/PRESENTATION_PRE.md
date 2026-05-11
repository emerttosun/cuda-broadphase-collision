# CMP674 Sunum Ön Taslağı

Bu dosya repoyu hızlıca sunuma dönüştürmek için hazırlanmış ön çalışma notudur. Amaç, final slaytları üretmeden önce projenin teknik hikayesini, güçlü sonuçlarını, demo akışını ve eksik kalan noktaları tek yerde toplamak.

## 1. Sunumun Ana Hikayesi

**Başlık önerisi:** CUDA ile 2B Broad-Phase Collision Detection: Brute Force, Uniform Grid, LBVH ve Spatial Hash Karşılaştırması

**Tek cümlelik tez:**
Brute-force çarpışma testi O(N²) candidate pair üretirken, broad-phase yöntemleri uzamsal düzeni kullanarak aday çift sayısını düşürür; CUDA üzerinde bu fark büyük N değerlerinde dramatik hızlanmaya dönüşür.

**Sunumun akışı:**

1. Problem: çok sayıda dairenin çarpışma çiftlerini bulmak.
2. Baseline: CPU brute force ve CUDA brute force.
3. Broad-phase fikri: tüm çiftleri değil, sadece yakın olabilecek çiftleri test etmek.
4. Uygulanan yöntemler: uniform grid, LBVH, spatial hash.
5. Doğruluk: parity testi ile CPU/CUDA yöntemlerinin collision count eşitliği.
6. Performans: `results/timings.csv` üzerinden total time, speedup ve candidate pair analizi.
7. Demo: raylib + CUDA/OpenGL interop görselleştirici.
8. Sınırlar ve gelecek iş: daha genel shape desteği, profiling, CUB, Morton grid, RT-core/OptiX.

## 2. Repo İncelemesi Özeti

Proje saf C + CUDA ağırlıklı bir CMake projesi olarak kurulmuş. Ana benchmark binary'si `collision_benchmark`, canlı demo binary'si `raylib_cuda_visualizer`, doğrulama binary'si ise `collision_parity_test`.

Ana klasörler:

| Klasör | Rol |
|---|---|
| `src/` | CPU/CUDA algoritmaları, benchmark sürücüsü, veri üretimi |
| `include/` | Ortak tipler, benchmark config, broadphase arayüzü |
| `tests/` | CPU/CUDA parity testi |
| `visualization/raylib-cuda-interop/` | raylib penceresi, CUDA simülasyon, OpenGL VBO interop |
| `scripts/` | Benchmark CSV'sinden grafik üretimi |
| `results/` | Ölçüm CSV çıktıları |
| `external/raylib/` | Git submodule olarak raylib |

Öne çıkan mimari parçalar:

| Parça | Dosyalar | Sunumda anlatılacak nokta |
|---|---|---|
| Ortak veri modeli | `include/Circle.h`, `include/CollisionResult.h` | Tüm yöntemler aynı circle ve result kontratını kullanıyor |
| Broadphase strategy | `include/Broadphase.h`, `src/Benchmark.c` | Yeni yöntem ekleme benchmark döngüsüne entegre edilmiş |
| CPU baseline | `src/CpuCollision.c` | Referans doğruluk ve speedup tabanı |
| CUDA brute force | `src/CudaBruteForce.cu` | GPU paralelliği var ama candidate count hâlâ O(N²) |
| Uniform grid | `src/CudaGrid.cu` | Cell key, sort, cell range, komşu hücre tarama |
| LBVH | `src/CudaLbvh.cu` | Morton code, radix tree, AABB traversal |
| Spatial hash | `src/CudaHash.cu` | Çok seviyeli hash/grid yaklaşımı |
| Visualizer | `visualization/raylib-cuda-interop/src/*` | CUDA simülasyon ve VBO'ya doğrudan yazma |
| Test | `tests/test_parity.c` | CPU, CUDA brute force, grid, LBVH ve hash collision count kıyası |

## 3. Algoritma Slaytları İçin İçerik

### Slide: Problem Tanımı

- Girdi: 2B düzlemde `N` adet daire: `(x, y, radius)`.
- Çıktı: çarpışan daire çiftlerinin sayısı.
- Naive yöntem: her `i < j` çifti için mesafe testi.
- Karmaşıklık: `N(N-1)/2`, yani O(N²).

Konuşmacı notu:
Bu proje narrow-phase collision çözümü değil; daire-daire testini kullanarak broad-phase yöntemlerinin aday çift sayısını nasıl azalttığını ölçüyor.

### Slide: CPU ve CUDA Brute Force

- CPU baseline doğru sonucu üretmek için referans.
- CUDA brute force her çifti paralel hesaplayarak CPU'ya göre kernel seviyesinde avantaj sağlar.
- Fakat candidate pair sayısı değişmez: hâlâ O(N²).
- Küçük N değerlerinde GPU transfer ve launch overhead'i CUDA brute force'u yavaş gösterebilir.

### Slide: Uniform Grid

Pipeline:

1. Her circle için cell id hesapla.
2. `cell_id` değerine göre indexleri sırala.
3. Her cell için başlangıç/bitiş aralığını çıkar.
4. Her obje için kendi hücresi ve komşu hücrelerde adayları tara.
5. Sadece aday çiftlerde gerçek circle overlap testi yap.

Sunumda vurgulanacak fikir:
Uniform grid'in kazancı GPU'nun ham paralelliğinden değil, candidate pair sayısını brute force'a göre ciddi düşürmesinden geliyor.

### Slide: LBVH

Pipeline:

1. Daire merkezlerini Morton code'a çevir.
2. Morton code'a göre sırala.
3. Linear BVH radix tree oluştur.
4. Leaf AABB'lerinden internal AABB'leri propagate et.
5. AABB traversal ile aday çiftleri bul.

Ne zaman anlamlı:
Uniform grid'in hücre yoğunlukları dengesizleştiği clustered senaryolarda daha adaptif bir uzamsal yapı olarak anlatılabilir.

### Slide: Spatial Hash

- Sabit sahne grid'inden farklı olarak cell koordinatlarını hash bucket'lara map eder.
- Radius profiline göre seviye seçimi yapılır.
- Sahne sınırı veya yoğunluk değişimi arttığında grid'e alternatif olarak konumlanır.

## 4. Benchmark Tasarımı

Benchmark matrisi:

| Boyut | Değerler |
|---|---|
| Object count | 1k, 5k, 10k, 50k, 100k |
| Distribution | uniform, clustered |
| Radius profile | narrow, mixed, extreme |
| Methods | CPU brute force, CUDA brute force, CUDA uniform grid, CUDA LBVH, CUDA hash |
| Grid cell sizes | 5, 10, 20, 40 |

CSV kolonları:

- `collision_count`: gerçek çarpışma sayısı.
- `candidate_pair_count`: yöntemin test ettiği aday çift sayısı.
- `kernel_time_ms`: CUDA kernel ölçümü.
- `total_time_ms`: transfer/allocation dahil uçtan uca süre.
- `speedup_vs_cpu`: aynı case için CPU total time'a göre hızlanma.
- `max_objects_in_cell`, `dense_cell_count`: grid/hash yoğunluk davranışı.

## 5. Ölçümden Çıkan Ön Bulgular

`results/timings.csv` içinde 240 satır benchmark sonucu var:

| Method | Satır sayısı |
|---|---:|
| CPU brute force | 30 |
| CUDA brute force | 30 |
| CUDA uniform grid | 120 |
| CUDA LBVH | 30 |
| CUDA hash | 30 |

Öne çıkan sonuç örnekleri:

| Case | Method | Total time | Speedup vs CPU | Candidate pairs |
|---|---|---:|---:|---:|
| 100k uniform narrow | CUDA uniform grid, cell=10 | 2.105 ms | 1524.29x | 4,437,462 |
| 100k uniform narrow | CUDA uniform grid, cell=20 | 2.614 ms | 1227.62x | 17,512,411 |
| 50k uniform narrow | CUDA uniform grid, cell=20 | 0.731 ms | 1084.41x | 4,372,998 |
| 100k clustered narrow | CUDA uniform grid, cell=5 | 5.475 ms | 907.87x | 452,381,569 |

Yorum:

- Büyük N değerlerinde uniform grid çok güçlü bir hızlanma sağlıyor.
- Cell size seçimi kritik: küçük cell candidate pair sayısını düşürür, fakat çok küçük cell overhead yaratabilir.
- Clustered dağılımda bazı cell'ler yoğunlaştığı için candidate pair sayısı keskin artıyor.
- CUDA brute force, algoritmik olarak O(N²) kaldığı için yalnızca paralelleştirme kazancı sunuyor; broad-phase kadar ölçeklenmiyor.

## 6. Doğruluk ve Güvenilirlik

Çalıştırılan doğrulama:

```powershell
ctest --test-dir build -C Release --output-on-failure
```

Sonuç:

```text
1/1 Test #1: collision_parity_test ............ Passed
100% tests passed, 0 tests failed out of 1
```

Sunum mesajı:
CPU brute force referans alınarak CUDA brute force, CUDA uniform grid, CUDA LBVH ve CUDA hash sonuçları parity testinde aynı collision count toleransı içinde doğrulanıyor.

## 7. Görselleştirici Demo Akışı

Demo binary:

```powershell
.\build\visualization\raylib-cuda-interop\Release\raylib_cuda_visualizer.exe 2500
```

Gösterilecek akış:

1. Başlangıçta uniform dağılım ve narrow radius profili göster.
2. `G` ile yöntemler arasında geç: CUDA brute force → CUDA uniform grid → CPU brute force → CUDA LBVH.
3. `C` ile clustered dağılıma geç; FPS ve collision/candidate etkisini anlat.
4. `V` ile radius profilini narrow → mixed → extreme yap.
5. `+` / `-` ile object count değiştir.
6. Kapanışta CUDA/OpenGL interop noktasını göster: particle vertex'leri CPU'ya geri kopyalanmadan VBO'ya yazılıyor.

Kısa demo anlatısı:
Mavi objeler çarpışmayanları, kırmızı objeler çarpışanları gösteriyor. Sol üst overlay'de yöntem, FPS, çarpışma sayısı ve GPU zamanı gibi canlı metrikler var.

## 8. Önerilen Slayt Planı

| # | Başlık | Amaç |
|---:|---|---|
| 1 | Problem ve Motivasyon | O(N²) maliyeti sezdir |
| 2 | Proje Mimarisine Genel Bakış | Benchmark, core library, visualizer, test |
| 3 | Veri Modeli ve Doğruluk Kriteri | Circle, CollisionResult, parity |
| 4 | CPU/CUDA Brute Force | Baseline ve limitleri |
| 5 | Uniform Grid Broad-Phase | Ana algoritma ve cell-size etkisi |
| 6 | LBVH | Morton + tree tabanlı alternatif |
| 7 | Spatial Hash | Grid'e esnek alternatif |
| 8 | Benchmark Matrisi | N, dağılım, radius, method |
| 9 | Sonuç 1: Total Time | Log-scale performans grafiği |
| 10 | Sonuç 2: Candidate Pairs | Brute force vs broad-phase farkı |
| 11 | Sonuç 3: Distribution Sensitivity | Uniform vs clustered |
| 12 | Live Visualizer | Demo ekranı ve kontroller |
| 13 | Limitations | Daire modeli, tek GPU, profiling eksikleri |
| 14 | Future Work | CUB, Morton grid, SoA, Nsight, RT-core |
| 15 | Sonuç | Broad-phase + CUDA ana çıkarımı |

## 9. Grafik Üretimi

Mevcut script:

```powershell
python scripts\plot_results.py
```

Beklenen çıktılar:

- `results/plots/*_total_time.png`
- `results/plots/*_speedup.png`
- `results/plots/*_candidate_pairs.png`
- `results/plots/grid_max_objects_in_cell.png`

Sunumda en faydalı grafikler:

1. `total_time_ms` vs `object_count`: yöntemlerin ölçeklenmesi.
2. `candidate_pair_count` vs `object_count`: broad-phase'in asıl etkisi.
3. `speedup_vs_cpu` vs `object_count`: anlatması kolay final metrik.
4. `max_objects_in_cell`: clustered dağılımın grid üzerindeki etkisi.

## 10. Eksikler ve Riskler

Sunumdan önce kontrol edilmesi iyi olur:

- README encoding'i terminalde mojibake görünüyor; final teslimde UTF-8 olarak doğrulanmalı.
- `scripts/plot_results.py` çalıştırılıp grafiklerin gerçekten üretildiği kontrol edilmeli.
- Benchmark sonuçlarının hangi GPU/CPU üzerinde alındığı sunumda belirtilmeli.
- `kernel_time_ms` ve `total_time_ms` farkı açık anlatılmalı; sadece kernel süresi üzerinden iddia kurulmamalı.
- Visualizer için ekran kaydı veya kısa GIF alınırsa canlı demo riskini azaltır.
- `git status` sandbox kullanıcısı nedeniyle okunamadı; final commit öncesi `safe.directory` ayarı veya normal kullanıcı terminaliyle kontrol edilmeli.

## 11. Literatür Bağlantısı

Kısa referans listesi:

- Le Grand, "Broad-Phase Collision Detection with CUDA", GPU Gems 3, Chapter 32.
- Simon Green, "Particle Simulation using CUDA".
- Tero Karras, "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees", HPG 2012.
- Liu et al., "Real-time Collision Culling of a Million Bodies on Graphics Processing Units", SIGGRAPH Asia 2010.
- Teschner et al., "Optimized Spatial Hashing for Collision Detection of Deformable Objects".

Sunumda literatür rolü:
Bu proje yeni bir collision detection teorisi önermekten çok, klasik GPU broad-phase tekniklerini aynı veri seti ve aynı doğruluk kontratı altında kıyaslayan uygulamalı bir performans çalışması olarak konumlandırılmalı.

## 12. Final Mesajı

Kapanış cümlesi önerisi:

> GPU brute force paralellik sağlar, fakat broad-phase algoritmik yükü azaltır. Bu projede en büyük kazanım ikisini birleştirmekten geliyor: CUDA paralelliği + uzamsal aday azaltma.

