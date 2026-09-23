#!/usr/bin/env python3
"""학습 데이터에서 좌·우 정규화 통계를 생성합니다.

실행:
    python export_normalization.py

명령행 parser는 사용하지 않습니다. 아래의 '사용자 설정' 경로만 수정합니다.
현재 train.py와 동일하게 각 측면에 대해 axis=(0, 2) 평균과 표준편차를
계산합니다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# =============================================================================
# 사용자 설정
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed_data"

LEFT_TRAIN_NPY_PATH = PROCESSED_DIR / "X_left_train.npy"
RIGHT_TRAIN_NPY_PATH = PROCESSED_DIR / "X_right_train.npy"
OUTPUT_PATH = BASE_DIR / "normalization.npz"

EPS = 1e-6
# =============================================================================


CHANNELS_PER_SIDE = 5


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} 파일을 찾을 수 없습니다:\n{path.resolve()}\n"
            "파일 상단의 경로 설정을 확인하십시오."
        )


def ensure_channel_first(x: np.ndarray, name: str) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError(f"{name}은 3차원이어야 합니다: {x.shape}")

    if x.shape[-1] == CHANNELS_PER_SIDE:
        x = np.transpose(x, (0, 2, 1))

    if x.shape[1] != CHANNELS_PER_SIDE:
        raise ValueError(f"{name}은 [N,5,T] 또는 [N,T,5]여야 합니다: {x.shape}")

    x = np.ascontiguousarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError(f"{name}에 NaN/Inf가 있습니다.")
    return x


def calculate_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    std = x.std(axis=(0, 2), keepdims=True).astype(np.float32)
    return mean, std


def main() -> None:
    require_file(LEFT_TRAIN_NPY_PATH, "좌측 train NPY")
    require_file(RIGHT_TRAIN_NPY_PATH, "우측 train NPY")

    left = ensure_channel_first(np.load(LEFT_TRAIN_NPY_PATH), "left_train")
    right = ensure_channel_first(np.load(RIGHT_TRAIN_NPY_PATH), "right_train")

    if left.shape[1:] != right.shape[1:]:
        raise ValueError(f"좌·우 feature shape가 다릅니다: {left.shape} vs {right.shape}")

    left_mean, left_std = calculate_stats(left)
    right_mean, right_std = calculate_stats(right)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_PATH,
        left_mean=left_mean,
        left_std=left_std,
        right_mean=right_mean,
        right_std=right_std,
        eps=np.float32(EPS),
    )

    print("\n정규화 통계 저장 완료")
    print(f"저장 경로        : {OUTPUT_PATH.resolve()}")
    print(f"left 입력 shape  : {left.shape}")
    print(f"right 입력 shape : {right.shape}")
    print(f"left_mean shape  : {left_mean.shape}")
    print(f"left_std shape   : {left_std.shape}")
    print(f"right_mean shape : {right_mean.shape}")
    print(f"right_std shape  : {right_std.shape}")
    print(f"eps              : {EPS}")


if __name__ == "__main__":
    main()
