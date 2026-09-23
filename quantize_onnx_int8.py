from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process


def ensure_channel_first(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected a 3-D array, got shape={x.shape}")
    if x.shape[-1] == 5:
        x = np.transpose(x, (0, 2, 1))
    if x.shape[1] != 5:
        raise ValueError(f"Expected [N, 5, T], got shape={x.shape}")
    return np.ascontiguousarray(x, dtype=np.float32)


class GRFCalibrationDataReader(CalibrationDataReader):
    def __init__(
        self,
        processed_dir: Path,
        normalization_path: Path,
        sample_count: int,
        seed: int,
    ):
        left_path = processed_dir / "X_left_train.npy"
        right_path = processed_dir / "X_right_train.npy"
        if not left_path.exists() or not right_path.exists():
            raise FileNotFoundError(
                f"Calibration arrays not found: {left_path}, {right_path}"
            )

        left = ensure_channel_first(np.load(left_path))
        right = ensure_channel_first(np.load(right_path))
        if left.shape != right.shape:
            raise ValueError(
                f"Left/right calibration shapes differ: {left.shape} vs {right.shape}"
            )

        stats = np.load(normalization_path)
        left = (left - stats["left_mean"]) / (stats["left_std"] + 1e-6)
        right = (right - stats["right_mean"]) / (stats["right_std"] + 1e-6)

        rng = np.random.default_rng(seed)
        count = min(int(sample_count), len(left))
        if count <= 0:
            raise ValueError("calibration-samples must be positive")
        indices = rng.choice(len(left), size=count, replace=False)

        self.left = np.ascontiguousarray(left[indices], dtype=np.float32)
        self.right = np.ascontiguousarray(right[indices], dtype=np.float32)
        self.position = 0

    def get_next(self) -> Dict[str, np.ndarray] | None:
        if self.position >= len(self.left):
            return None

        index = self.position
        self.position += 1
        return {
            "sensor_left": self.left[index : index + 1],
            "sensor_right": self.right[index : index + 1],
        }

    def rewind(self) -> None:
        self.position = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Static INT8 QDQ quantization for the GRF ONNX model."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("deploy/student_qkd_fp32.onnx"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("deploy/student_qkd_int8.onnx"),
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("processed_data"),
    )
    parser.add_argument(
        "--normalization",
        type=Path,
        default=Path("deploy/normalization.npz"),
    )
    parser.add_argument("--calibration-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--calibration-method",
        choices=["minmax", "entropy", "percentile"],
        default="minmax",
    )
    parser.add_argument(
        "--keep-preprocessed",
        action="store_true",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input ONNX model not found: {args.input}")
    if not args.normalization.exists():
        raise FileNotFoundError(
            f"Normalization file not found: {args.normalization}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    preprocessed = args.output.with_name(
        f"{args.output.stem}_preprocessed.onnx"
    )

    quant_pre_process(
        str(args.input),
        str(preprocessed),
        skip_optimization=False,
    )

    reader = GRFCalibrationDataReader(
        processed_dir=args.processed_dir,
        normalization_path=args.normalization,
        sample_count=args.calibration_samples,
        seed=args.seed,
    )

    calibration_methods = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }

    quantize_static(
        model_input=str(preprocessed),
        model_output=str(args.output),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=calibration_methods[args.calibration_method],
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
        },
    )

    if not args.keep_preprocessed and preprocessed.exists():
        preprocessed.unlink()

    fp32_mib = args.input.stat().st_size / (1024**2)
    int8_mib = args.output.stat().st_size / (1024**2)
    print(f"FP32 ONNX: {fp32_mib:.2f} MiB")
    print(f"INT8 ONNX: {int8_mib:.2f} MiB")
    print(f"Saved: {args.output}")
    print(
        "Conv/MatMul/Gemm are quantization candidates. LayerNorm, Softmax, "
        "residual arithmetic, and unsupported nodes remain floating point."
    )


if __name__ == "__main__":
    main()
