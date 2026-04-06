# CR-Project

## 프로젝트 개요
- 위성 영상 클라우드 제거를 위한 모듈화된 학습 프레임워크
- modules/ 디렉토리에 재사용 가능한 컴포넌트 구현 중
- uv 패키지 매니저 사용, Python 3.x

## 코드 스타일
- Loss 함수는 modules/ 하위에 독립 모듈로 구현
- 다른 모듈(모델, 옵티마이저 등)과 호환되도록 일반화된 인터페이스 유지
- main.py의 build_loss() 패턴을 따를 것

## 주의사항
- main.py, main.ipynb는 main 브랜치에서만 수정
- 개인 실험은 tmp_main.py 사용

## 현재 진행 중인 작업 (2026-04-06)

### Ablation: CrossModalBlock ×1 vs ×2
- **브랜치**: `feat/ablation_xmodal` (`feat/loss_fn`에서 분기)
- **목적**: SAR 정보 주입 지점을 1곳(body1 뒤)에서 2곳(body1+body2 뒤)으로 늘렸을 때 성능 차이 검증
- **변경 파일**:
  - `modules/model/ACA_CRNet.py` — `cross_modal_num_blocks` 파라미터 추가 (기본값 1, 기존 동작 유지)
  - `tmp_main_xmodal1.py` — ×1 실험 (출력: `artifacts/module1_mae_xmodal1_20ep`)
  - `tmp_main_xmodal2.py` — ×2 실험 (출력: `artifacts/module1_mae_xmodal2_20ep`)
- **실험 설정**: MAE(L1) loss, AdamW(lr=1e-4), train=512, val=64, test=64, batch=4, epochs=20
- **할 일**: Colab에서 두 스크립트 실행 후 val metrics 비교