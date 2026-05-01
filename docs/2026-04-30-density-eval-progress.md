# Density Evaluation Progress Report

작성일: 2026-04-30

## 1. 목적

이번 실험의 1차 목표는 복원 성능 자체를 바로 비교하는 것이 아니라, `density map`이 실제 구름 영향 영역을 얼마나 잘 반영하는지 먼저 확인하는 것이다.

여기서 density는 단순 보조값이 아니라, 모델 안에서 `SAR 개입 강도를 조절하는 gate` 역할을 한다. 따라서 이 실험은 엄밀한 의미의 물리적 `cloud density estimator` 비교라기보다, `cloud-aware SAR injection gate` 비교에 가깝다.

## 2. 비교할 density 설계

### `cosine`

- 현재 baseline
- SAR feature와 optical feature를 같은 차원으로 투영한 뒤
- cosine similarity 기반으로 density를 계산

### `cosine_prior`

- `cosine` 결과에 optical prior를 추가
- optical prior는 Sentinel-2 cloud-score heuristic 계열을 PyTorch로 옮긴 보조 신호

의도:
- 맑은 영역 오탐 감소
- 얇은 구름, haze 보조

## 2-1. Optical Prior 설명과 출처

`cosine_prior`에서 쓰는 prior는 학습된 cloud detector가 아니라, `Sentinel-2 optical band`만 보고 구름스러운 정도를 점수화하는 `rule-based heuristic`이다.

핵심 아이디어:
- 구름은 대체로 밝다
- 얇은 구름과 cirrus는 aerosol/cirrus 관련 밴드에서 힌트가 드러난다
- 구름은 snow, 밝은 지면, 일부 표면 반사와 구분되어야 한다

현재 코드에서는 다음 신호를 사용한다.
- `blue`
- `aerosol`
- `aerosol + cirrus`
- `red + green + blue`
- `NDMI = (nir - swir1) / (nir + swir1)`
- `NDSI = (green - swir1) / (green + swir1)`

구현 방식:
- 각 규칙을 `0~1` 점수로 rescale
- 여러 규칙을 `min` 방식으로 순차 결합
- 마지막에 pooling으로 약간 smoothing

이 prior의 가장 직접적인 참고 근거는 다음 문헌이다.

1. Schmitt, M., Hughes, L. H., Qiu, C., and Zhu, X. X. (2019).
   *Aggregating Cloud-Free Sentinel-2 Images with Google Earth Engine*.
   ISPRS Annals.
   https://doi.org/10.5194/isprs-annals-IV-2-W7-145-2019

이 논문은 Sentinel-2용 cloud score를 설명하면서 다음 원리를 사용한다.
- clouds are bright
- clouds are moist
- clouds are not snow

그리고 실제 규칙으로 `blue`, `aerosol`, `cirrus+aerosol`, `RGB`, `NDMI`, `NDSI`를 순차적으로 사용한다. 현재 코드의 prior는 이 계열과 매우 유사한 구조를 따른다.

물리적 해석의 배경은 다음 문헌이 뒷받침한다.

2. Hollstein, A., Segl, K., Guanter, L., Brell, M., and Enesco, M. (2016).
   *Ready-to-Use Methods for the Detection of Clouds, Cirrus, Snow, Shadow, Water and Clear Sky Pixels in Sentinel-2 MSI Images*.
   Remote Sensing, 8(8), 666.
   https://doi.org/10.3390/rs8080666

이 문헌은 Sentinel-2 밴드 중 특히 `cirrus band (B10)`와 visible/SWIR 계열이 cloud, cirrus, snow, shadow 구분에 중요한 이유를 설명한다.

정리:
- `prior`는 새로 학습한 detector가 아님
- Sentinel-2 cloud-score heuristic 계열을 참고한 optical-only cloud-likeness map임
- 현재 프로젝트에서는 이를 `SAR injection gate`를 안정화하는 보조 신호로 사용함

## 3. 구현 상태

구현은 완료된 상태다.

추가된 구성:
- `tmp_main.py`에 `--density-mode` 추가
- density estimator 모듈 분리
- density-only 평가 스크립트 추가
- JSON, summary plot, example image 저장 기능 추가

주요 코드 위치:
- `modules/model/cafm/density/cosine.py`
- `modules/model/cafm/density/prior.py`
- `modules/model/cafm/density/factory.py`
- `modules/metrics/density_eval.py`
- `scripts/eval_density.py`

## 4. 현재 아키텍처 해석

현재 구조는 다음 흐름으로 이해하는 것이 가장 정확하다.

1. 입력 `OPT + SAR`에서 density를 먼저 계산한다.
2. 본선 복원 경로는 ACA-CRNet body를 따라 진행한다.
3. 중간의 `C1`, `C2`에서 SAR와 cross-attention을 수행한다.
4. 이때 density를 사용해 `어디에서 SAR를 얼마나 주입할지` 조절한다.

따라서 현재 density map은 다음 의미에 더 가깝다.

- 구름의 절대 밀도
- 보다, `optical을 얼마나 덜 믿어야 하는지`
- 또는 `SAR를 얼마나 더 주입해야 하는지`

즉 해석상으로는 `cloud density map`보다 `SAR guidance gate` 또는 `optical unreliability map`에 가깝다.

## 5. 평가 방법

### 기본 원칙

density ground truth는 없기 때문에, 직접 정답 비교는 불가능하다. 따라서 `proxy-based evaluation`으로 density를 비교한다.

### proxy 정의

각 밴드에서 구름 영향도를 다음과 같이 본다.

```text
P_c(x) = |cloudy_c(x) - target_c(x)|
```

중점 밴드:
- Blue
- Red
- NIR
- Cirrus
- SWIR1

추가로 구름 민감 밴드를 가중합한 weighted proxy도 사용한다.

가중치:
- blue: 0.30
- cirrus: 0.30
- swir1: 0.20
- red: 0.10
- nir: 0.10

### 저장 지표

- `corr_blue`
- `corr_red`
- `corr_nir`
- `corr_cirrus`
- `corr_swir1`
- `corr_weighted`
- `clear_mean_d`
- `thin_mean_d`
- `thick_mean_d`
- `top10_proxy`
- `top20_proxy`

해석 기준:
- `corr_*`, `corr_weighted`: 높을수록 좋음
- `clear_mean_d`: 낮을수록 좋음
- `thick_mean_d`: 높을수록 좋음
- `top10_proxy`, `top20_proxy`: 높을수록 좋음

즉, 좋은 density는:
- 맑은 곳에서는 낮고
- 실제 구름 영향이 큰 곳에서는 높아야 한다

## 6. 왜 5 epoch 비교를 먼저 하는가

이번 5 epoch 실험은 최종 결론용이 아니라 `screening` 용도다.

목적:
- 세 가지 density 아이디어의 초기 경향 비교
- 명백히 불리한 후보 제거
- 이후 장기 학습 대상을 압축

해석 원칙:
- 큰 차이는 의미 있게 볼 수 있음
- 작은 차이는 아직 성급하게 결론 내리면 안 됨

## 7. 현재 진행 상황

현재 아티팩트 기준 진행 상태:

- `cosine`: 5 epoch 학습 완료
- `cosine_prior`: 5 epoch 학습 완료
- `cosine_prior`: 20 epoch까지 추가 학습 후 density 비교 완료
- `cosine`: 장기 baseline checkpoint(`cafm_l1`)로 density 비교 완료

확인된 결과 파일:
- `artifacts/density_runs/cosine_5ep/best.pt`
- `artifacts/density_runs/cosine_5ep/history.png`
- `artifacts/density_runs/cosine_prior_5ep/best.pt`
- `artifacts/density_runs/cosine_prior_5ep/history.png`

중요한 해석:
- `best.pt`가 `epoch 2`라고 해서 학습이 2 epoch에서 멈춘 것은 아님
- 의미는 `5 epoch까지 학습한 뒤, 검증 기준 최고 성능이 epoch 2였다`는 것

## 8. 실행 명령

### 학습

`cosine`

```bash
uv run --no-sync python tmp_main.py \
  --density-mode cosine \
  --loss-type l1 \
  --output-dir artifacts/density_runs/cosine_5ep \
  --batch-size 2 \
  --max-epochs 5 \
  --save-every 1
```

`cosine_prior`

```bash
uv run --no-sync python tmp_main.py \
  --density-mode cosine_prior \
  --loss-type l1 \
  --output-dir artifacts/density_runs/cosine_prior_5ep \
  --batch-size 2 \
  --max-epochs 5 \
  --save-every 1
```

### 중단 후 재개

```bash
uv run --no-sync python tmp_main.py \
  --density-mode cosine_prior \
  --loss-type l1 \
  --output-dir artifacts/density_runs/cosine_prior_5ep \
  --batch-size 2 \
  --max-epochs 5 \
  --resume artifacts/density_runs/cosine_prior_5ep/best.pt
```

주의:
- 재개는 배치 중간부터가 아니라 마지막 저장 checkpoint 기준이다
- `--save-every 1`을 쓰면 에폭 단위 재개가 더 안전하다

### density-only 비교

두 모드의 `best.pt`가 모두 준비되었으므로, 다음 명령으로 바로 비교할 수 있다.

```bash
python scripts/eval_density.py \
  --split validation \
  --max-samples 512 \
  --batch-size 2 \
  --checkpoint cosine=artifacts/density_runs/cosine_5ep/best.pt \
  --checkpoint cosine_prior=artifacts/density_runs/cosine_prior_5ep/best.pt
```

20 epoch 비교 시 사용한 명령:

```bash
python scripts/eval_density.py \
  --split validation \
  --max-samples 512 \
  --batch-size 2 \
  --checkpoint cosine=/home/2021112028/cr-project/artifacts/cafm_l1/periodic/epoch_090.pt \
  --checkpoint cosine_prior=/home/2021112028/cr-project/artifacts/density_runs/cosine_prior_5ep/periodic/epoch_020.pt
```

## 9. 결과 저장 위치

density 비교 결과 저장 경로:

- `artifacts/density_compare/validation/samples_512/metrics.json`
- `artifacts/density_compare/validation/samples_512/summary.png`
- `artifacts/density_compare/validation/samples_512/examples/`

학습 결과 저장 경로:

- `artifacts/density_runs/cosine_5ep/`
- `artifacts/density_runs/cosine_prior_5ep/`
## 10. 1차 비교 결과

`eval_density.py`를 validation 512 samples 기준으로 실행한 결과는 다음과 같다.

실행 설정:
- split: `validation`
- samples: `512`
- batch-size: `2`
- checkpoints:
  - `artifacts/density_runs/cosine_5ep/best.pt`
  - `artifacts/density_runs/cosine_prior_5ep/best.pt`

핵심 수치:

- `cosine`
  - `corr_weighted`: `0.180`
  - `clear_mean_d`: `0.429`
  - `thick_mean_d`: `0.445`
  - `top10_proxy`: `0.481`

- `cosine_prior`
  - `corr_weighted`: `0.320`
  - `clear_mean_d`: `0.462`
  - `thick_mean_d`: `0.473`
  - `top10_proxy`: `0.737`

결론:

- 현재 1등 후보는 `cosine_prior`
- `cosine`은 baseline으로는 동작하지만 분리력이 더 약함

해석:

- `cosine_prior`는 proxy와의 상관이 가장 높고, 구름 영향이 큰 영역에서 density를 가장 잘 올림
- `cosine`은 전체 구조는 어느 정도 따라가지만 density가 더 noisy하고 구분력이 약함

## 11. 시각화 확인 결과

summary plot과 example image를 함께 확인한 결과, 수치와 시각화가 같은 결론을 가리켰다.

- `cosine`
  - proxy의 큰 구조는 일부 따라감
  - 다만 density가 전반적으로 noisy하고 공간 분리가 약함

- `cosine_prior`
  - proxy의 밝은 구조와 density high-response 영역이 가장 잘 맞음
  - spatial pattern도 가장 자연스럽고 해석 가능성이 높음

실무적 판단:

- 현재 long-run 후보는 `cosine_prior`
- `cosine`은 baseline 유지

## 12. 20 epoch 비교 결과

5 epoch screening 이후, 장기 학습에서도 gate가 유지되는지 보기 위해 추가 비교를 진행했다.

실행 설정:
- split: `validation`
- samples: `512`
- batch-size: `2`
- checkpoints:
  - `cosine`: `/home/2021112028/cr-project/artifacts/cafm_l1/periodic/epoch_090.pt`
  - `cosine_prior`: `/home/2021112028/cr-project/artifacts/density_runs/cosine_prior_5ep/periodic/epoch_020.pt`

핵심 수치:

- `cosine`
  - `corr_weighted`: `0.0029`
  - `clear_mean_d`: `1.000`
  - `thin_mean_d`: `1.000`
  - `thick_mean_d`: `1.000`
  - `top10_proxy`: `0.376`
  - `top20_proxy`: `0.376`

- `cosine_prior`
  - `corr_weighted`: `-0.0629`
  - `clear_mean_d`: `0.165`
  - `thin_mean_d`: `0.164`
  - `thick_mean_d`: `0.161`
  - `top10_proxy`: `0.302`
  - `top20_proxy`: `0.313`

추가 해석용 구분 지표:

- `thick-clear gap = thick_mean_d - clear_mean_d`
- `top10-clear gap = top10_proxy - clear_mean_d`

계산 결과:

- `cosine`
  - `thick-clear gap = 0.000`
  - `top10-clear gap = -0.624`

- `cosine_prior`
  - `thick-clear gap = -0.004`
  - `top10-clear gap = +0.137`

해석:

- `cosine`은 사실상 전 영역을 `1`로 보내는 `all-ones collapse` 상태다.
- 즉 맑은 영역과 구름 영향 영역을 구분하지 못하며, gate로서는 실패다.
- `cosine_prior`는 `cosine`보다는 덜 극단적이지만, 전체 상관이 음수이고 `thick_mean_d < clear_mean_d`라서 여전히 좋은 gate라고 보기 어렵다.
- 결론적으로 현재 학습 설정에서는 `20 epoch` 시점에 두 방식 모두 안정적인 gate를 유지하지 못했다.

## 13. 현재 해석

현재까지의 중요한 결론은 다음과 같다.

1. `5 epoch` 기준 screening에서는 `cosine_prior`가 가장 유망했다.
2. 그러나 `20 epoch` 비교에서는 `cosine`, `cosine_prior` 모두 gate quality가 무너졌다.
3. 따라서 현재 구조와 학습 설정에서는 `장기 학습이 자동으로 더 좋은 gate를 만든다`고 보기 어렵다.

구조적으로는 다음과 같이 해석할 수 있다.

- `cosine`: 장기 학습 중 "어차피 SAR를 전부 넣자" 방향으로 포화되었을 가능성
- `cosine_prior`: 장기 학습 중 "전반적으로 gate를 낮게 두자" 방향으로 수축되었을 가능성

즉 현재 문제는 단순 성능 저하라기보다, `gate collapse / saturation` 가능성이 높다.

## 14. 주의사항

- `batch-size 4`는 validation/test에서 GPU OOM 가능성이 큼
- 이유는 validation/test가 crop 없이 `256x256` full resolution으로 돌기 때문
- ACA-CRNet의 attention 메모리 사용량이 커서 현재 비교 실험은 `batch-size 2`가 안전함

## 15. 현재 결론과 다음 액션

현재 기준의 올바른 순서는 다음과 같다.

1. `5 epoch screening` 결과와 `20 epoch collapse` 결과를 함께 인정한다.
2. 현재 상태에서 바로 장기 실험을 더 늘리기보다, `gate collapse` 원인을 먼저 점검한다.
3. 다음 단계에서는 density regularization 또는 clear/thick separation 유도 장치가 필요한지 검토한다.
4. 이후 수정된 설정으로 다시 `5 epoch -> 20 epoch` 추적 비교를 수행한다.

내일 이어서 볼 핵심 포인트:

- 왜 `cosine`이 장기 학습에서 `all-ones`로 포화되는지
- 왜 `cosine_prior`가 장기 학습에서 전체적으로 낮게 수축되는지
- density/gate에 직접적인 안정화 제약이 필요한지
- density 평가와 실제 복원 성능이 얼마나 일치하는지

현재 메모의 핵심 요약:
- 구현은 끝났다
- 2개 density 아이디어의 5 epoch screening과 20 epoch 비교까지 완료했다
- 현재 비교 대상은 `cloud density estimator`라기보다 `SAR injection gate`이다
- `5 epoch`에서는 `cosine_prior`가 가장 유망했다
- 하지만 `20 epoch`에서는 두 방법 모두 stable gate를 유지하지 못했다
- 다음 핵심 작업은 `gate collapse` 원인 분석과 안정화 방안 검토다
