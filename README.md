# cr-project

## Protected Files

`main.py` 와 `main.ipynb` 는 공용 학습 러너 파일이라서 `main` 브랜치에서만 커밋되도록 훅을 둡니다.

## Local Temp Runner

공용 러너를 건드리지 않고 개인 실험을 하려면 `tmp_main_base.py` 를 복사해서 `tmp_main.py` 로 만들어 쓰면 됩니다.

```bash
cp tmp_main_base.py tmp_main.py
uv run python tmp_main.py --help
```

- `tmp_main_base.py` 는 저장소에 남는 템플릿 파일입니다
- `tmp_main.py` 는 각자 수정해서 쓰는 로컬 파일이고 `.gitignore` 에 들어 있습니다
- `tmp_main.py` 에서는 `build_model()`, `build_optimizer()` 를 구현하고 필요하면 `build_loss()`, `build_metrics()` 도 덮어쓰면 됩니다
- 실행 자체는 공용 `main.py` 가 담당하므로 학습 루프와 결과 저장 흐름은 동일하게 유지됩니다

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
