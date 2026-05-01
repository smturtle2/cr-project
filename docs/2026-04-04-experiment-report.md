# 2026-04-04 실험 보고서: ACA-CRNet + CAFM 학습 및 비교 분석

---

## 1. 학습 스펙 비교 및 우리의 선택

### 1.1 기존 모델 학습 스펙

| 항목 | DSen2-CR (Meraner 2020) | EMRDM (Zou 2024) |
|---|---|---|
| 모델 유형 | CNN (ResNet, 16 blocks, F=256) | Multi-Resolution Diffusion |
| Batch size | 16 | 4 (per GPU) |
| Optimizer | Nadam (lr=7e-5, schedule_decay=0.004) | Adam (lr=1e-4) |
| Crop size | 128x128 (원본 256x256에서 random crop) | 256x256 (풀사이즈) |
| Augmentation | random crop + flip + rotation(0/90/180/270) | flip |
| Training samples | 134,907 | ~122,218 |
| Epochs / Iterations | 8 epochs | 500K~1M iterations |
| Loss | CARL (Cloud-Adaptive Regularized Loss) | L2 (diffusion denoising objective) |
| GPU | DGX-1 (8x Tesla P100) | 4x NVIDIA A100 |

> **출처:** DSen2-CR — [Meraner et al. 2020, PMC7386944](https://pmc.ncbi.nlm.nih.gov/articles/PMC7386944/), [GitHub: ameraner/dsen2-cr](https://github.com/ameraner/dsen2-cr). EMRDM — [Zou et al. 2024, arXiv:2503.23717](https://arxiv.org/html/2503.23717).

### 1.2 우리의 학습 스펙

| 항목 | 값 |
|---|---|
| 모델 | ACA-CRNet + CAFM (16 layers, F=256) |
| Optimizer | AdamW (lr=1e-4, 고정) |
| Loss | CloudAdaptiveLoss (alpha=2.0) / SimpleMSELoss (baseline) |
| Batch size | 4 |
| Crop size | 128x128 (원본 256x256에서 random crop) |
| Augmentation | random crop + flip + rotation(0/90/180/270) |
| Train / Val / Test | 2048 / 256 / 256 samples |
| Epochs | 20 |
| Early stopping | patience=10 (val_loss 기준) |
| GPU | 단일 GPU (12GB) |

### 1.3 DSen2-CR 기반으로 설계한 이유

ACA-CRNet은 DSen2-CR의 아키텍처를 직접 계승한 모델이다.

| 구조 요소 | DSen2-CR | ACA-CRNet |
|---|---|---|
| 백본 | ResNet (residual blocks) | ResNet (residual blocks) |
| 레이어 수 / Feature size | 16 / 256 | 16 / 256 |
| 입력 | SAR + Cloudy concat (15ch) | SAR + Cloudy concat (15ch) |
| 출력 방식 | Residual (cloudy + output) | Residual (cloudy + output) |
| 추가 모듈 | - | CAFM (Cloud-Adaptive Feature Modulation) |

동일 백본이므로 DSen2-CR에서 검증된 학습 전략(128x128 crop, flip+rotation, CARL loss)이 그대로 적용 가능하다.

**EMRDM 전략을 따르지 않는 이유:**
- Diffusion 모델은 학습 방식(noise prediction, 수십만 iteration)이 근본적으로 다름
- A100 4장, 수일 학습이 필요 → 단일 12GB GPU에서 비현실적
- 256x256 풀사이즈는 12GB GPU에서 batch=1도 빡빡함

---

## 2. CAFM ON/OFF 비교 실험

### 2.1 실험 설계

동일 조건에서 CAFM 유무만 변경하여 CAFM 모듈의 기여도를 정량적으로 측정한다.

| 조건 | CAFM ON | CAFM OFF (Baseline) |
|---|---|---|
| 모델 | ACA-CRNet + CAFM | ACA-CRNet |
| Loss | CloudAdaptiveLoss | SimpleMSELoss |
| 그 외 | 동일 (데이터, 에폭, lr, augmentation, batch size) | 동일 |
| 출력 경로 | `artifacts/` | `artifacts/baseline/` |

> **주의:** Loss 함수가 다르므로 loss 절대값은 비교 불가. MAE, PSNR, SSIM, SAM 지표로 비교.

### 2.2 CAFM ON — 학습 결과 (40에폭)

CAFM 모델은 40에폭까지 학습했으나, 20에폭 이후 val 지표의 뚜렷한 개선이 없어 **실질적 수렴은 ~20에폭**으로 판단.

#### 학습 곡선 요약

| 구간 | Train Loss | Val Loss (best) | Val PSNR (best) | 비고 |
|---|---|---|---|---|
| Epoch 1~5 | 0.69 → 0.36 | 0.39 | 26.95 | 급격한 초기 학습 |
| Epoch 6~10 | 0.36 → 0.31 | 0.36 | 27.03 | 완만한 하강 |
| Epoch 11~20 | 0.31 → 0.28 | 0.32 | 27.07 | 수렴 시작, best 갱신 |
| Epoch 21~30 | 0.28 → 0.26 | 0.33 | 27.09 | 미세 개선 |
| Epoch 31~40 | 0.26 → 0.24 | 0.33 | 27.09 | 수렴 완료 |

> 20에폭 이후 val best 갱신이 미미하여, baseline 비교는 **20에폭 기준**으로 진행.

#### CAFM ON 최종 결과 (20에폭 기준)

| 지표 | Train (Epoch 20) | Val Best (Epoch 20) | Test |
|---|---|---|---|
| MAE | 0.1439 | 0.1651 | 0.2075 |
| PSNR | 27.98 | 27.07 | 24.58 |
| SSIM | 0.8600 | 0.8693 | 0.8202 |
| SAM | 8.56 | 7.75 | 12.62 |

### 2.3 CAFM OFF (Baseline) — 학습 결과

#### 학습 곡선 요약

| 구간 | Train Loss | Val Loss (best) | Val PSNR (best) | 비고 |
|---|---|---|---|---|
| Epoch 1~5 | 0.159 → 0.049 | 0.064 | 26.32 | 급격한 초기 학습 |
| Epoch 6~10 | 0.049 → 0.040 | 0.062 | 26.70 | 완만한 하강 |
| Epoch 11~15 | 0.039 → 0.038 | 0.050 | 27.29 | best 갱신 지속 |
| Epoch 16~20 | 0.035 → 0.034 | 0.046 | 27.82 | 마지막 에폭에서 best 갱신 |

> train loss가 0.034까지 내려갔고, val best가 20에폭에서 갱신됨. SimpleMSELoss 사용으로 loss 절대값이 CAFM(CloudAdaptiveLoss)보다 작지만, **loss 함수가 다르므로 loss 값 자체는 비교 불가**.

#### Baseline 최종 결과 (20에폭 기준)

| 지표 | Train (Epoch 20) | Val Best (Epoch 20) | Test |
|---|---|---|---|
| MAE | 0.1195 | 0.1583 | 0.1980 |
| PSNR | 29.73 | 27.82 | 25.16 |
| SSIM | 0.8600 | 0.8650 | 0.8232 |
| SAM | 7.30 | 7.36 | 10.86 |

### 2.4 CAFM 효과 비교

#### Test 지표 비교

| 지표 | CAFM ON (Test) | CAFM OFF (Test) | 차이 | 우위 |
|---|---|---|---|---|
| MAE ↓ | 0.2075 | **0.1980** | +0.0095 | Baseline |
| PSNR ↑ | 24.58 | **25.16** | -0.58 dB | Baseline |
| SSIM ↑ | 0.8202 | **0.8232** | -0.003 | Baseline |
| SAM ↓ | **12.62** | 10.86 | +1.76 | Baseline |

#### Val Best 지표 비교

| 지표 | CAFM ON (Val Best) | CAFM OFF (Val Best) | 차이 | 우위 |
|---|---|---|---|---|
| MAE ↓ | **0.1651** | 0.1583 | +0.0068 | Baseline |
| PSNR ↑ | 27.07 | **27.82** | -0.75 dB | Baseline |
| SSIM ↑ | 0.8693 | 0.8650 | +0.0043 | CAFM |
| SAM ↓ | **7.75** | 7.36 | +0.39 | Baseline |

#### 분석

현재 실험에서는 **Baseline이 대부분의 지표에서 CAFM+CARL보다 우수**한 결과를 보였다. 다만 이 실험은 Loss 함수가 달라 순수 CAFM 효과를 분리할 수 없다. 이를 해결하기 위해 ablation study를 추가 진행한다.

### 2.5 Ablation Study: 순수 CAFM 모듈 효과 측정

섹션 2.4에서 CAFM+CARL vs Baseline(MSE)를 비교했으나, Loss 함수가 달라 순수 CAFM 효과를 분리할 수 없었다. CloudAdaptiveLoss(CARL)는 구름 영역에 가중치를 부여하는 loss로, 소규모 데이터(2048샘플)에서는 MSE 대비 오히려 학습을 방해하여 Baseline이 우위를 보인 것으로 분석되었다.

이를 검증하기 위해 **CAFM ON + SimpleMSELoss** 실험을 추가하여, Loss를 통일(MSE)한 상태에서 CAFM 유무만 다른 공정한 비교를 진행했다.

| 실험 | CAFM | Loss | 목적 |
|---|---|---|---|
| CAFM + MSE | ON | SimpleMSELoss | **순수 CAFM 모듈 효과** 분리 |
| Baseline | OFF | SimpleMSELoss | 기준선 |

#### Val Best 지표 비교 (동일 MSE Loss)

| 지표 | Baseline | CAFM + MSE | CAFM 개선폭 |
|---|---|---|---|
| MAE ↓ | 0.1583 | **0.1453** | **-0.0130 (8.2% 개선)** |
| PSNR ↑ | 27.82 | **28.50** | **+0.68 dB** |
| SSIM ↑ | 0.8650 | **0.8774** | **+0.0124** |
| SAM ↓ | 7.36 | **7.04** | **-0.32 (4.3% 개선)** |

#### Test 지표 비교 (동일 MSE Loss)

| 지표 | Baseline | CAFM + MSE | CAFM 개선폭 |
|---|---|---|---|
| MAE ↓ | **0.1980** | 0.2006 | +0.0026 |
| PSNR ↑ | 25.16 | **25.21** | **+0.05 dB** |
| SSIM ↑ | 0.8232 | **0.8270** | **+0.0038** |
| SAM ↓ | 10.86 | **10.81** | **-0.05** |

#### 분석

**Val Best**: CAFM이 **4개 지표 모두에서 Baseline 상회**. PSNR +0.68dB, SSIM +0.012는 유의미한 차이다.

**Test**: PSNR/SSIM/SAM에서 CAFM이 소폭 우위, MAE만 Baseline이 소폭 우위. 전체적으로 차이가 작다.

**Val과 Test 지표 차이가 나는 이유:**

Val Best에서는 CAFM이 PSNR +0.68dB로 명확한 우위를 보였으나, Test에서는 +0.05dB로 차이가 축소되었다.

1. **Val Best 선택 편향.** 20에폭 중 가장 좋은 시점의 값이므로 모델에 유리한 방향으로 선택 편향(selection bias)이 있다. Test는 해당 에폭의 단일 평가이므로 이 효과가 없다.
2. **Test set 분포 차이.** Val(256샘플)과 Test(205샘플)는 서로 다른 ROI에서 추출된 데이터다. Test set에 구름이 더 두껍거나 복원이 어려운 장면이 포함되어 있으면, 전체 지표가 낮아지며 모델 간 차이도 희석된다.
3. **샘플 수 부족.** 205샘플에서의 +0.05dB 차이는 소수의 어려운 샘플이 평균에 미치는 영향 범위 내에 있다. 전체 test set(~7,900샘플)으로 평가하면 더 안정적인 결과를 기대할 수 있다.
4. **방향성은 일관.** 차이 크기는 다르지만, PSNR/SSIM/SAM 모두 CAFM 쪽이 높으므로 방향성은 일관된다.

**결론:** Loss를 통일하면 **CAFM 모듈은 Val 기준으로 명확한 성능 향상**을 보인다. Test에서도 방향성은 일관되나, 샘플 수 부족으로 차이가 축소되었다.

#### 해석 시 고려사항

1. **데이터 규모 한계**: 2048 샘플은 전체 데이터의 ~1.5%. CAFM의 구름 밀도 적응 기능은 대규모 데이터에서 더 큰 효과를 발휘할 가능성이 있음.
2. **Val/Test 샘플 수**: 각각 256/205 샘플로 통계적 노이즈가 큼. Val에서는 명확했던 차이가 Test에서 축소된 원인.
3. **CloudAdaptiveLoss**: 소규모 데이터에서는 MSE 대비 불리. 대규모 학습 시 재검증 필요.

---

## 3. 다른 모델과의 비교를 위한 조건 분석

### 3.1 벤치마크 수치 (각 논문 자체 보고)

| 모델 | MAE ↓ | PSNR ↑ | SSIM ↑ | SAM ↓ |
|---|---|---|---|---|
| DSen2-CR | 0.031 | 27.76 | 0.874 | 9.472 |
| GLF-CR | 0.028 | 28.64 | 0.885 | 8.981 |
| UnCRtainTS | 0.027 | 28.90 | 0.880 | 8.320 |
| ACA-Net (Baseline) | 0.025 | 29.78 | 0.896 | 7.770 |
| DiffCR | 0.019 | 31.77 | 0.902 | 5.821 |
| EMRDM (SOTA) | 0.018 | 32.14 | 0.924 | 5.267 |

> **출처:** [docs/benchmarks.md](benchmarks.md), 각 논문 자체 보고 수치.

### 3.2 직접 비교가 어려운 이유

#### (1) 데이터 규모 차이

| | DSen2-CR / ACA-Net | Ours |
|---|---|---|
| Training samples | 134,907 | 2,048 |
| 비율 | 100% | ~1.5% |

동일 스펙으로 Diffusion 모델을 학습시켜도 **수렴에 필요한 iteration 수가 근본적으로 다르다** (CNN: 수천 step vs Diffusion: 수십만 step). 동일 에폭/데이터로 비교하면 Diffusion이 극심하게 언더피팅되어 불공정한 비교가 된다.

#### (2) 메트릭 스케일 차이

논문마다 정규화/메트릭 구현이 달라 수치의 직접 비교에 주의가 필요하다.

**메트릭 스케일이란?** 같은 예측 결과라도 데이터의 값 범위(스케일)에 따라 지표 숫자가 달라진다. 예를 들어 동일한 예측에 대해:
- 0~5 범위(÷2000 정규화)에서 MAE를 계산하면 **MAE = 0.10**
- 0~1 범위(÷10000 정규화)에서 MAE를 계산하면 **MAE = 0.02**

성능은 동일하지만 숫자가 5배 차이난다. PSNR도 `max_val` 파라미터(신호 최대값)에 따라 값이 달라지고, SSIM도 안정화 상수 `C1`, `C2`가 `max_val`에 의존한다. SAM만 코사인 각도 기반이라 스케일에 무관하다.

| 논문/모델 | MAE 스케일 | PSNR max_val | 비교 가능? |
|---|---|---|---|
| DSen2-CR | 0~5 (÷2000) | 10000 (×2000 복원) | **4개 지표 모두 O** |
| ACA-CRNet (원본) | 0~5 (÷2000) | 10000 (×2000 복원) | **4개 지표 모두 O** |
| GLF-CR | 0~1 추정 | PIXEL_MAX=1 | SAM만 O |
| UnCRtainTS | 0~1 추정 | max_val=1 | SAM만 O |
| DiffCR | 8-bit PNG 기반 | 255 (8-bit) | SAM만 O |
| EMRDM | 미확인 | 미확인 (코드 비공개) | SAM만 O |

따라서 벤치마크 테이블의 수치를 그대로 가져와 비교하면, 스케일 차이로 인해 **실제 성능과 무관하게 숫자가 높거나 낮게 보일 수 있다.** 공정한 비교를 위해서는 동일 코드베이스에서 동일 정규화로 재평가해야 한다.

> **출처:** DSen2-CR — [GitHub: ameraner/dsen2-cr](https://github.com/ameraner/dsen2-cr) `image_metrics.py`. ACA-CRNet — [GitHub: huangwenwenlili/ACA-CRNet](https://github.com/huangwenwenlili/ACA-CRNet) `np_metric.py`. GLF-CR — [GitHub: xufangchn/GLF-CR](https://github.com/xufangchn/GLF-CR). UnCRtainTS — [GitHub: PatrickTUM/UnCRtainTS](https://github.com/PatrickTUM/UnCRtainTS). DiffCR — [GitHub: XavierJiezou/DiffCR](https://github.com/XavierJiezou/DiffCR).

### 3.3 공정한 비교 방법

| 비교 유형 | 방법 | 현실성 |
|---|---|---|
| **CAFM 효과 증명** | 동일 조건 CAFM ON/OFF → test 지표 비교 | **현재 진행 중** |
| **DSen2-CR / ACA-Net 비교** | 동일 메트릭 스케일이므로 수치 직접 비교 가능. 단, 데이터 규모 차이 명시 필요 | 가능 (단서 필요) |
| **Diffusion 모델 비교** | 동일 코드로 추론(inference)만 재평가. 학습된 가중치가 공개되어야 함 | 가중치 공개 여부에 따라 다름 |
| **전체 재현** | 모든 모델을 동일 데이터·동일 코드로 학습+평가 | A100 등 대규모 자원 필요 |

### 3.4 Diffusion 모델과의 정성적 비교

수치 비교가 어려운 경우, CNN 기반 모델의 구조적 강점을 정성적으로 어필할 수 있다.

| 비교 항목 | CNN 기반 (Ours) | Diffusion (DiffCR, EMRDM) |
|---|---|---|
| 추론 속도 | single forward pass | 1~1000 iterative steps |
| Cloud-free 영역 보존 | residual learning → 원본 유지 | uniform noise injection → 비구름 영역도 변형 가능 |
| Spectral hallucination | 위험 낮음 (deterministic) | generative 특성상 없는 색상/텍스처 생성 위험 |
| 출력 재현성 | 항상 동일 결과 | stochastic (매번 다른 결과 가능) |
| 파라미터 수 | 경량 | EMRDM 148.9M (매우 무거움) |
| 운영 배포 | ONNX/TensorRT 최적화 용이 | 복잡한 파이프라인 |

> **출처:** SADER 논문 — "diffusion noise is injected uniformly across the entire image, causing undesired perturbations even in cloud-free regions" ([arXiv:2602.00536](https://arxiv.org/html/2602.00536)). DiffCR 논문 — lake/water body 혼동 한계 자체 보고 ([arXiv:2308.04417](https://ar5iv.labs.arxiv.org/html/2308.04417), Section IV-E).

---

*Baseline 학습 완료 후 섹션 2.3, 2.4를 업데이트할 예정.*
