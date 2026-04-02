from __future__ import annotations

import main as shared_main
import torch
from torch import nn


# 이 파일은 그대로 실행하는 용도가 아니라 개인 작업용 템플릿이다.
# 보통은 아래처럼 복사해서 `tmp_main.py`를 만든 뒤 그 파일을 수정해서 사용한다.
#
#   cp tmp_main_base.py tmp_main.py
#   uv run python tmp_main.py --help
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
    return shared_main.build_loss()


def build_metrics() -> dict[str, shared_main.MetricFn]:
    # metric도 기본 구현을 그대로 쓰게 두고,
    # 개인 실험에서 필요할 때만 `tmp_main.py`에서 수정하면 된다.
    return shared_main.build_metrics()


def main() -> None:
    # 공용 러너는 `main.py`에 두고,
    # 이 템플릿은 교체 가능한 build 함수만 덮어쓴 뒤 그대로 실행을 위임한다.
    shared_main.build_model = build_model
    shared_main.build_optimizer = build_optimizer
    shared_main.build_loss = build_loss
    shared_main.build_metrics = build_metrics
    shared_main.main()


if __name__ == "__main__":
    main()
