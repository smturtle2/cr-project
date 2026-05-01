# SEN12MS-CR Benchmark

## Comparison with State-of-the-Art Methods

| Model | MAE↓ | PSNR↑ | SSIM↑ | SAM↓ |
|---|---|---|---|---|
| McGAN | 0.048 | 25.14 | 0.744 | 15.676 |
| SAR-Opt-cGAN | 0.043 | 25.59 | 0.764 | 15.494 |
| SAR2OPT | 0.042 | 25.87 | 0.793 | 14.788 |
| SpA GAN | 0.045 | 24.78 | 0.754 | 18.085 |
| Simulation-Fusion GAN | 0.045 | 24.73 | 0.701 | 16.633 |
| DSen2-CR | 0.031 | 27.76 | 0.874 | 9.472 |
| GLF-CR | 0.028 | 28.64 | 0.885 | 8.981 |
| UnCRtainTS L2 | 0.027 | 28.90 | 0.880 | 8.320 |
| ACA-Net (Baseline) | 0.025 | 29.78 | 0.896 | 7.770 |
| DiffCR | 0.019 | 31.77 | 0.902 | 5.821 |
| **EMRDM** | **0.018** | **32.14** | **0.924** | **5.267** |
| Ours — Baseline (MSE, 2048 train) | 0.1980 | 25.16 | 0.8232 | 10.86 |
| Ours — CAFM (MSE, 2048 train) | 0.2006 | 25.21 | 0.8270 | 10.81 |
| Ours — SDI+CAFM (MSE, 4096 train) | 0.1668 | 26.68 | 0.8417 | 10.50 |
| Ours — SDI+CAFM (L1, 4096 train) | 0.1598 | 26.88 | 0.8657 | 9.928 |
| Ours — Baseline (L1, 4096 train) | 0.1610 | 26.93 | 0.8584 | 9.533 |
| Ours — CAFM (L1, 4096 train) | 0.1591 | 26.95 | 0.8634 | 9.900 |
| Ours — CAFM (L1, 4096 train, 100ep) | 0.1509 | 27.68 | 0.8783 | 9.018 |

## Ours — Val / Test Breakdown (MSE Loss, 20 epochs)

### 2048 train samples (batch 4, crop 128×128)

**Val Best**

| Model | MAE↓ | PSNR↑ | SSIM↑ | SAM↓ |
|---|---|---|---|---|
| Baseline | 0.1583 | 27.82 | 0.8650 | 7.36 |
| CAFM | 0.1453 | 28.50 | 0.8774 | 7.04 |

**Test**

| Model | MAE↓ | PSNR↑ | SSIM↑ | SAM↓ |
|---|---|---|---|---|
| Baseline | 0.1980 | 25.16 | 0.8232 | 10.86 |
| CAFM | 0.2006 | 25.21 | 0.8270 | 10.81 |

### 4096 train samples (batch 4, crop 128×128)

**Val Best**

| Model | Loss | epoch | MAE↓ | PSNR↑ | SSIM↑ | SAM↓ |
|---|---|---|---|---|---|---|
| Baseline | L1 | 7 | 0.2024 | 26.37 | 0.8418 | 9.008 |
| CAFM | L1 | 2 | 0.2027 | 26.34 | 0.8391 | 8.858 |
| SDI+CAFM | MSE | 5 | 0.2080 | 25.86 | 0.8366 | 9.251 |
| SDI+CAFM | L1 | 2 | 0.1962 | 26.64 | 0.8382 | 8.503 |

**Test**

| Model | Loss | MAE↓ | PSNR↑ | SSIM↑ | SAM↓ |
|---|---|---|---|---|---|
| Baseline | L1 | 0.1610 | 26.93 | 0.8584 | 9.533 |
| CAFM | L1 | 0.1591 | 26.95 | 0.8634 | 9.900 |
| SDI+CAFM | MSE | 0.1668 | 26.68 | 0.8417 | 10.50 |
| SDI+CAFM | L1 | 0.1598 | 26.88 | 0.8657 | 9.928 |

### 4096 train samples, 100 epochs (batch 4, crop 128×128)

**Val Best (epoch 79)**

| Model | Loss | epoch | MAE↓ | PSNR↑ | SSIM↑ | SAM↓ |
|---|---|---|---|---|---|---|
| CAFM | L1 | 79 | 0.1751 | 27.88 | 0.8914 | 6.849 |

**Test**

| Model | Loss | MAE↓ | PSNR↑ | SSIM↑ | SAM↓ |
|---|---|---|---|---|---|
| CAFM | L1 | 0.1509 | 27.68 | 0.8783 | 9.018 |

**비고 (100 epochs)**:
- Best epoch=79. 20ep 대비 Val PSNR 26.43→27.88 (+1.45dB), Test PSNR 26.95→27.68 (+0.73dB).
- Train PSNR 33.94 vs Val PSNR 27.22 (gap ~6.7dB) → 과적합 뚜렷하나 val best는 계속 갱신 중.
- Epoch 40 이후 val loss 0.175~0.200 범위에서 진동, 수렴 근접.

**비고 (20 epochs)**:
- 2048 train → 4096 train로 데이터 2배 증가.
- MSE: 과적합 심함 (train PSNR 30.81 vs val PSNR 25.86, gap 5 dB). Best epoch=5.
- L1: best epoch=2로 더 빠르게 peak. Test에서 MSE보다 PSNR +0.2dB, SSIM +0.024, SAM -0.57 개선.
- Baseline L1: best epoch=7. 과적합 뚜렷 (train MAE 0.100 vs val MAE 0.202).
- CAFM L1: best epoch=2. Test에서 Baseline 대비 MAE -0.002, PSNR +0.02, SSIM +0.005. 차이 미미.
- L1 3종 비교 (Baseline / CAFM / SDI+CAFM): Test PSNR 26.93 / 26.95 / 26.88, SSIM 0.858 / 0.863 / 0.866. CAFM 단독 효과는 미미하나 SDI+CAFM에서 SSIM이 가장 높음.
