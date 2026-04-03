from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import main as shared_main
from pathlib import Path

import torch
from torch import nn


# 공용 main의 기본 build 함수는 monkey patch 전에 미리 잡아 둔다.
# 그래야 아래에서 shared_main.build_loss/build_metrics를 덮어쓴 뒤에도
# 기본 구현을 재귀 없이 다시 호출할 수 있다.
_DEFAULT_BUILD_LOSS = shared_main.build_loss
_DEFAULT_BUILD_METRICS = shared_main.build_metrics


# 이 파일은 그대로 실행하는 용도가 아니라 개인 작업용 템플릿이다.
# 보통은 아래처럼 복사해서 `tmp_main.py`를 만든 뒤 그 파일을 수정해서 사용한다.
#
#   cp tmp_main_base.py tmp_main.py
#   uv run python tmp_main.py
#
# `tmp_main.py`는 gitignore에 들어 있으므로 공용 `main.py`를 건드리지 않고
# 각자 실험용 모델/옵티마이저/loss/metric을 자유롭게 연결할 수 있다.


def build_model() -> nn.Module:
    # 개인 실험에서 사용할 실제 모델을 여기서 반환한다.
    # cr-train 규약상 `forward(sar, cloudy)` 시그니처만 맞추면 된다.
    raise NotImplementedError("tmp_main.py에서 build_model()을 구현하세요.")


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    # build_model()이 만든 모델 파라미터를 받아 optimizer를 구성한다.
    raise NotImplementedError("tmp_main.py에서 build_optimizer()를 구현하세요.")


def build_loss() -> shared_main.LossFn:
    # loss는 기본 `main.py` 구현을 그대로 쓰거나,
    # 필요하면 `tmp_main.py`에서 이 함수를 직접 덮어써서 바꿔도 된다.
    # 부분 수정이 필요하면 `_DEFAULT_BUILD_LOSS()`를 호출해서 기본 구현을 재사용하면 된다.
    return _DEFAULT_BUILD_LOSS()


def build_metrics() -> dict[str, shared_main.MetricFn]:
    # metric도 기본 구현을 그대로 쓰게 두고,
    # 개인 실험에서 필요할 때만 `tmp_main.py`에서 수정하면 된다.
    # 기본 metric에 몇 개를 더 얹고 싶다면 `_DEFAULT_BUILD_METRICS()`를 먼저 받아서 확장하면 된다.
    return _DEFAULT_BUILD_METRICS()


@contextmanager
def use_local_builds() -> Iterator[None]:
    # 공용 러너는 `main.py`에 두고,
    # 이 템플릿은 교체 가능한 build 함수만 잠시 덮어쓴 뒤 실행을 위임한다.
    # 실행이 끝난 뒤 원래 함수를 복구해 두면 같은 프로세스에서 `main`을 다시 import해서 쓸 때
    # 개인용 build 함수가 전역 상태처럼 남아 버리는 문제를 막을 수 있다.
    overrides = {
        "build_model": build_model,
        "build_optimizer": build_optimizer,
        "build_loss": build_loss,
        "build_metrics": build_metrics,
    }
    originals = {name: getattr(shared_main, name) for name in overrides}

    for name, override in overrides.items():
        setattr(shared_main, name, override)

    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(shared_main, name, original)


def main() -> None:
    with use_local_builds():
        shared_main.main(
            batch_size=8,
            seed=42,
            max_epochs=10,
            train_max_samples=16384,
            val_max_samples=2048,
            test_max_samples=2048,
            output_dir=Path("artifacts"),
            resume=None,
            run_test=False,
            num_examples=4,
            example_stage="val",
        )


if __name__ == "__main__":
    main()
