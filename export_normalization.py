#!/usr/bin/env python3
"""Generate left and right normalization statistics from the training data.

Usage:
    python export_normalization.py

No command-line parser is used. Edit only the paths in 'User settings' below.
For each side, compute the mean and standard deviation over axis=(0, 2),
consistent with the current implementation in train.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# =============================================================================
# User settings
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
            f"{description} file not found:\n{path.resolve()}\n"
            "Check the path settings at the top of this file."
        )


def ensure_channel_first(x: np.ndarray, name: str) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError(f"{name} must be three-dimensional: {x.shape}")

    if x.shape[-1] == CHANNELS_PER_SIDE:
        x = np.transpose(x, (0, 2, 1))

    if x.shape[1] != CHANNELS_PER_SIDE:
        raise ValueError(f"{name} must have shape [N,5,T] or [N,T,5]: {x.shape}")

    x = np.ascontiguousarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contains NaN/Inf values.")
    return x


def calculate_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    std = x.std(axis=(0, 2), keepdims=True).astype(np.float32)
    return mean, std


def main() -> None:
    require_file(LEFT_TRAIN_NPY_PATH, "Left training NPY")
    require_file(RIGHT_TRAIN_NPY_PATH, "Right training NPY")

    left = ensure_channel_first(np.load(LEFT_TRAIN_NPY_PATH), "left_train")
    right = ensure_channel_first(np.load(RIGHT_TRAIN_NPY_PATH), "right_train")

    if left.shape[1:] != right.shape[1:]:
        raise ValueError(f"Left/right feature shapes differ: {left.shape} vs {right.shape}")

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

    print("\nNormalization statistics saved")
    print(f"Output path      : {OUTPUT_PATH.resolve()}")
    print(f"left input shape : {left.shape}")
    print(f"right input shape: {right.shape}")
    print(f"left_mean shape  : {left_mean.shape}")
    print(f"left_std shape   : {left_std.shape}")
    print(f"right_mean shape : {right_mean.shape}")
    print(f"right_std shape  : {right_std.shape}")
    print(f"eps              : {EPS}")


if __name__ == "__main__":
    main()
