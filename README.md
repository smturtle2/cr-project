# cr-project

## Protected Files

`main.py` 와 `main.ipynb` 는 공용 학습 러너 파일이라서 `main` 브랜치에서만 커밋되도록 훅을 둡니다.

Feature 브랜치에서는 `main.py` 를 직접 건드리지 말고 로컬 전용 `tmp_main.py` 를 만들어 그 파일로 실험합니다.
`tmp_main.py` 는 `.gitignore` 에 들어 있어서 커밋되지 않습니다.

### 한 번만 설정

각 로컬 클론에서 아래 명령을 한 번 실행하면 저장소 안의 훅을 사용합니다.

```bash
git config core.hooksPath .githooks
```

### 동작 방식

- 현재 브랜치가 `main` 이면 `main.py`, `main.ipynb` 커밋 가능
- 현재 브랜치가 `main` 이 아니면 두 파일이 staging 되어 있을 때 `git commit` 실패
- `modules/...` 같은 feature 코드 파일은 그대로 커밋 가능
- feature 브랜치에서는 `cp main.py tmp_main.py` 로 복사한 뒤 `tmp_main.py` 에서 러너를 바꿔 씀

### 막혔을 때

`main.py` 나 `main.ipynb` 를 feature 브랜치에서 실수로 수정했다면:

1. 변경을 되돌리거나
2. 별도 브랜치가 아니라 `main` 에서 작업하거나
3. 정말 공통 변경이라면 `main` 에 먼저 반영한 뒤 feature 브랜치를 다시 `rebase` 하면 됩니다

`tmp_main.py` 가 없으면 아래처럼 새로 만들면 됩니다.

```bash
cp main.py tmp_main.py
python tmp_main.py --help
```

### 주의

이 방식은 로컬 커밋 시점을 막는 용도입니다.

- 팀원마다 `git config core.hooksPath .githooks` 를 한 번씩 해야 합니다
- `git commit --no-verify` 로는 우회할 수 있습니다
