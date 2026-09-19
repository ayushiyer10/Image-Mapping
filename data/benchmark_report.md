# Surface Map Estimation Benchmark Report

**Comparison**: Trivial Flat Baseline vs. Classical DIP (Sobel) vs. Deep Learning (MiDaS_small)
**Evaluated Samples**: 166 | **Wall-clock Time**: 416.34 seconds

## Summary Scorecard (Normal Map Metrics)

| Source | Estimator | Mean MAE (°) | Median MAE (°) | % < 11.25° | % < 22.5° | PSNR (dB) | SSIM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Diode** | Trivial Flat Baseline | 42.54° | 41.98° | 5.97% | 24.79% | 13.21 | 0.6993 |
| **Diode** | Classical DIP (Sobel) | 44.15° | 43.47° | 5.94% | 22.46% | 12.88 | 0.5069 |
| **Diode** | Deep Learning (MiDaS) | 42.64° | 41.76° | 6.06% | 24.64% | 13.21 | 0.695 |
| **Polyhaven** | Trivial Flat Baseline | 10.92° | 10.61° | 65.5% | 86.45% | 26.12 | 0.5456 |
| **Polyhaven** | Classical DIP (Sobel) | 34.2° | 33.7° | 9.14% | 29.91% | 14.72 | 0.1187 |
| **Polyhaven** | Deep Learning (MiDaS) | 11.13° | 10.7° | 64.64% | 86.25% | 25.48 | 0.5441 |
| **Overall** | Trivial Flat Baseline | 39.47° | 40.82° | 11.75% | 30.77% | 14.46 | 0.6844 |
| **Overall** | Classical DIP (Sobel) | 43.18° | 42.6° | 6.25% | 23.19% | 13.06 | 0.4693 |
| **Overall** | Deep Learning (MiDaS) | 39.59° | 40.58° | 11.74% | 30.61% | 14.4 | 0.6804 |

## Roughness & Height Maps (PBR & Synthetic)

| Source | Roughness PSNR | Roughness SSIM | Height PSNR | Height SSIM |
| :--- | :---: | :---: | :---: | :---: |
| **Diode** | N/A | N/A | N/A | N/A |
| **Polyhaven** | 16.24 dB | 0.2729 | 11.52 dB | 0.2725 |
| **Overall** | 16.24 dB | 0.2729 | 11.52 dB | 0.2725 |