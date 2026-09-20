# Surface Map Estimation Benchmark Report

**Comparison**: Trivial Flat Baseline vs. Classical DIP (Sobel) vs. Deep Learning (MiDaS_small)
**Evaluated Samples**: 516 | **Wall-clock Time**: 1625.52 seconds

## Summary Scorecard (Normal Map Metrics)

| Source | Estimator | Mean MAE (°) | Median MAE (°) | % < 11.25° | % < 22.5° | PSNR (dB) | SSIM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Diode** | Trivial Flat Baseline | 51.23° | 50.78° | 5.1% | 17.67% | 12.36 | 0.6913 |
| **Diode** | Classical DIP (Sobel) | 53.04° | 52.83° | 4.79% | 16.39% | 12.05 | 0.4708 |
| **Diode** | Deep Learning (MiDaS) | 51.38° | 50.97° | 5.08% | 17.61% | 12.36 | 0.6861 |
| **Polyhaven** | Trivial Flat Baseline | 10.92° | 10.61° | 65.5% | 86.45% | 26.12 | 0.5456 |
| **Polyhaven** | Classical DIP (Sobel) | 34.2° | 33.7° | 9.14% | 29.91% | 14.72 | 0.1187 |
| **Polyhaven** | Deep Learning (MiDaS) | 11.13° | 10.7° | 64.64% | 86.25% | 25.48 | 0.5441 |
| **Overall** | Trivial Flat Baseline | 49.98° | 49.84° | 6.97% | 19.8% | 12.78 | 0.6868 |
| **Overall** | Classical DIP (Sobel) | 52.46° | 51.98° | 4.93% | 16.81% | 12.13 | 0.4599 |
| **Overall** | Deep Learning (MiDaS) | 50.13° | 50.14° | 6.93% | 19.74% | 12.77 | 0.6817 |

## Roughness & Height Maps (PBR & Synthetic)

| Source | Roughness PSNR | Roughness SSIM | Height PSNR | Height SSIM |
| :--- | :---: | :---: | :---: | :---: |
| **Diode** | N/A | N/A | N/A | N/A |
| **Polyhaven** | 16.24 dB | 0.2729 | 11.52 dB | 0.2725 |
| **Overall** | 16.24 dB | 0.2729 | 11.52 dB | 0.2725 |