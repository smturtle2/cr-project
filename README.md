# cr-project

## Dependency Policy

서버 호환성에 따라 학습 의존성을 둘 중 하나로 설치합니다.

- `torch 2.6.x`가 필요한 서버: `uv sync --extra torch26`
- 최신 `torch`가 필요한 서버: `uv sync --extra latest`
- 위 두 명령은 모두 `cr-train`과 해당 `torch` 버전을 함께 설치합니다
- 현재 `latest` extra는 `torch>=2.11` 기준으로 잠기며, `uv lock --upgrade-package torch`로 최신 stable을 다시 반영할 수 있습니다
- bare `uv sync`만 실행하면 학습 의존성인 `cr-train`, `torch`가 설치되지 않습니다
- `torch26`와 `latest`는 동시에 선택할 수 없습니다
- 버전 확인은 `uv run python -c "import torch; print(torch.__version__)"`로 확인합니다

## Protected Files

`main.py` 와 `main.ipynb` 는 공용 학습 러너 파일이라서 `main` 브랜치에서만 커밋되도록 훅을 둡니다.

## Local Temp Runner

공용 러너를 건드리지 않고 개인 실험을 하려면 `tmp_main_base.py` 를 복사해서 `tmp_main.py` 로 만들어 쓰면 됩니다.

```bash
cp tmp_main_base.py tmp_main.py
uv run python tmp_main.py
```

- `tmp_main_base.py` 는 저장소에 남는 템플릿 파일입니다
- `tmp_main.py` 는 각자 수정해서 쓰는 로컬 파일이고 `.gitignore` 에 들어 있습니다
- `tmp_main.py` 에서는 `build_model()`, `build_optimizer()` 를 구현하고 필요하면 `build_loss()`, `build_metrics()` 도 덮어쓰면 됩니다
- 학습 설정은 CLI parse 대신 `main(...)` 호출 인자로 직접 넘깁니다
- 실행 자체는 공용 `main.py` 가 담당하므로 학습 루프와 결과 저장 흐름은 동일하게 유지됩니다
- 노트북/임시 러너에서도 `torch`가 필요하면 먼저 `uv sync --extra torch26` 또는 `uv sync --extra latest`로 환경을 맞춥니다
- 최신 `cr-train` 기준으로 데이터 로딩은 항상 block-cache streaming 경로를 사용합니다
- 전체 split을 쓰려면 `train_max_samples=None`, `val_max_samples=None`, `test_max_samples=None` 처럼 sample limit을 `None` 으로 넘기면 됩니다

### 한 번만 설정

각 로컬 클론에서 아래 명령을 한 번 실행하면 저장소 안의 훅을 사용합니다.

```bash
git config core.hooksPath .githooks
```

### 동작 방식

- 현재 브랜치가 `main` 이면 `main.py`, `main.ipynb` 커밋 가능
- 현재 브랜치가 `main` 이 아니면 두 파일이 staging 되어 있을 때 `git commit` 실패
- `modules/...` 같은 feature 코드 파일은 그대로 커밋 가능

### 주의

이 방식은 로컬 커밋 시점을 막는 용도입니다.

- 팀원마다 `git config core.hooksPath .githooks` 를 한 번씩 해야 합니다
- `git commit --no-verify` 로는 우회할 수 있습니다
