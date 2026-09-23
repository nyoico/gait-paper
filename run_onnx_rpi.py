from __future__ import annotations

import argparse
import csv
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR / "student_qkd_int8.onnx").exists():
    DEFAULT_ASSET_DIR = SCRIPT_DIR
else:
    DEFAULT_ASSET_DIR = SCRIPT_DIR / "deploy"

if (SCRIPT_DIR / "X_left_test.npy").exists():
    DEFAULT_DATA_DIR = SCRIPT_DIR
else:
    DEFAULT_DATA_DIR = SCRIPT_DIR / "processed_data"

DEFAULT_TARGETS_PATH = DEFAULT_DATA_DIR / "y_left_test.npy"
if not DEFAULT_TARGETS_PATH.exists():
    DEFAULT_TARGETS_PATH = None


# 5-class 그룹 순서는 train.py의 COARSE_CLASSES와 같게 맞춘다.
GROUP_CLASSES = ["HC", "H", "K", "A", "C"]
FINE_TO_GROUP = {
    "HC": "HC",
    "H_P": "H",
    "H_C": "H",
    "H_F": "H",
    "K_P": "K",
    "K_F": "K",
    "K_R": "K",
    "A_F": "A",
    "A_R": "A",
    "A_L": "A",
    "C_F": "C",
    "C_A": "C",
}


def ensure_channel_first(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        if x.shape[-1] == 5:
            x = x.T
        x = x[None, ...]
    elif x.ndim == 3 and x.shape[-1] == 5:
        x = np.transpose(x, (0, 2, 1))

    if x.ndim != 3 or x.shape[1] != 5:
        raise ValueError(f"Expected [N, 5, T] or [N, T, 5], got {x.shape}")
    return np.ascontiguousarray(x, dtype=np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def load_label_mapping(path: Path) -> dict[int, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Expected a non-empty index-to-label JSON object: {path}")

    labels = {int(index): str(label) for index, label in raw.items()}
    expected_indices = set(range(len(labels)))
    if set(labels) != expected_indices:
        raise ValueError(
            "Label indices must be contiguous from 0 to "
            f"{len(labels) - 1}, got {sorted(labels)}"
        )
    if len(set(labels.values())) != len(labels):
        raise ValueError("Label names must be unique.")
    return labels


def normalize_inputs(
    left: np.ndarray,
    right: np.ndarray,
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    required_keys = {"left_mean", "left_std", "right_mean", "right_std"}
    with np.load(path) as stats:
        missing = required_keys.difference(stats.files)
        if missing:
            raise KeyError(
                f"Missing normalization values in {path}: {sorted(missing)}"
            )
        left_mean = stats["left_mean"]
        left_std = stats["left_std"]
        right_mean = stats["right_mean"]
        right_std = stats["right_std"]

    try:
        left_normalized = (left - left_mean) / (left_std + 1e-6)
        right_normalized = (right - right_mean) / (right_std + 1e-6)
    except ValueError as error:
        raise ValueError(
            "Normalization statistics cannot be broadcast to the input shapes: "
            f"left={left.shape}, right={right.shape}"
        ) from error

    if not np.all(np.isfinite(left_normalized)) or not np.all(
        np.isfinite(right_normalized)
    ):
        raise ValueError("Normalized inputs contain NaN or infinity.")

    return (
        np.ascontiguousarray(left_normalized, dtype=np.float32),
        np.ascontiguousarray(right_normalized, dtype=np.float32),
    )


def target_to_index(value: object, label_to_index: dict[str, int]) -> int:
    if isinstance(value, bytes):
        value = value.decode("utf-8")

    label = str(value)
    if label in label_to_index:
        return label_to_index[label]

    try:
        index = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Unknown target label: {value!r}") from error

    if isinstance(value, (float, np.floating)) and not float(value).is_integer():
        raise ValueError(f"Target class index must be an integer, got {value!r}")
    if not 0 <= index < len(label_to_index):
        raise ValueError(f"Target class index is out of range: {index}")
    return index


def load_targets(
    path: Path,
    label_to_index: dict[str, int],
    expected_count: int,
    sample_indices: np.ndarray,
) -> np.ndarray:
    raw = np.asarray(np.load(path, allow_pickle=True)).reshape(-1)
    if len(raw) != expected_count:
        raise ValueError(
            f"Target count differs from dataset: {len(raw)} vs {expected_count}"
        )

    targets = np.asarray(
        [target_to_index(value, label_to_index) for value in raw],
        dtype=np.int64,
    )
    return targets[sample_indices]


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def classification_report(
    matrix: np.ndarray,
    class_names: list[str],
) -> dict[str, dict[str, float | int | None]]:
    report: dict[str, dict[str, float | int | None]] = {}
    for index, name in enumerate(class_names):
        true_positive = int(matrix[index, index])
        support = int(matrix[index].sum())
        predicted = int(matrix[:, index].sum())
        precision = true_positive / predicted if predicted else None
        recall = true_positive / support if support else None
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision is not None
            and recall is not None
            and precision + recall > 0
            else None
        )
        report[name] = {
            "support": support,
            "predicted": predicted,
            "correct": true_positive,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return report


def fine_to_group_indices(
    indices: np.ndarray,
    index_to_label: dict[int, str],
) -> np.ndarray:
    group_to_index = {
        label: index for index, label in enumerate(GROUP_CLASSES)
    }
    grouped: list[int] = []
    for index in indices:
        fine_label = index_to_label[int(index)]
        if fine_label not in FINE_TO_GROUP:
            raise KeyError(f"No group mapping defined for fine label: {fine_label}")
        grouped.append(group_to_index[FINE_TO_GROUP[fine_label]])
    return np.asarray(grouped, dtype=np.int64)


def run_dataset_inference(
    session: ort.InferenceSession,
    left: np.ndarray,
    right: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, list[float]]:
    logits_batches: list[np.ndarray] = []
    batch_times_ms: list[float] = []
    class_count: int | None = None

    for start_index in range(0, len(left), batch_size):
        end_index = min(start_index + batch_size, len(left))
        inputs = {
            "sensor_left": np.ascontiguousarray(left[start_index:end_index]),
            "sensor_right": np.ascontiguousarray(right[start_index:end_index]),
        }
        start_time = time.perf_counter_ns()
        logits = np.asarray(session.run(["logits"], inputs)[0])
        batch_times_ms.append(
            (time.perf_counter_ns() - start_time) / 1_000_000.0
        )

        expected_batch = end_index - start_index
        if logits.ndim != 2 or logits.shape[0] != expected_batch:
            raise ValueError(
                "Expected ONNX logits shaped [batch, classes], got "
                f"{logits.shape} for batch size {expected_batch}"
            )
        if not np.all(np.isfinite(logits)):
            raise ValueError(
                "ONNX logits contain NaN or infinity in batch "
                f"{start_index}:{end_index}."
            )
        if class_count is None:
            class_count = int(logits.shape[1])
        elif logits.shape[1] != class_count:
            raise ValueError("ONNX output class count changed between batches.")
        logits_batches.append(logits)

    if not logits_batches:
        raise ValueError("The dataset contains no samples.")
    return np.concatenate(logits_batches, axis=0), batch_times_ms


def benchmark_single_sample(
    session: ort.InferenceSession,
    left: np.ndarray,
    right: np.ndarray,
    warmup: int,
    repeat: int,
) -> list[float]:
    inputs = {
        "sensor_left": np.ascontiguousarray(left[:1]),
        "sensor_right": np.ascontiguousarray(right[:1]),
    }
    for _ in range(warmup):
        session.run(["logits"], inputs)

    times_ms: list[float] = []
    for _ in range(repeat):
        start_time = time.perf_counter_ns()
        session.run(["logits"], inputs)
        times_ms.append((time.perf_counter_ns() - start_time) / 1_000_000.0)
    return times_ms


def save_predictions(
    path: Path,
    sample_indices: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    index_to_label: dict[int, str],
    targets: np.ndarray | None,
) -> None:
    class_names = [index_to_label[index] for index in range(len(index_to_label))]
    fieldnames = [
        "sample_index",
        "prediction_index",
        "prediction_label",
        "prediction_group",
        "confidence",
    ]
    if targets is not None:
        fieldnames.extend(["true_index", "true_label", "true_group", "correct"])
    probability_fields = [f"probability_{name}" for name in class_names]
    fieldnames.extend(probability_fields)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, sample_index in enumerate(sample_indices):
            prediction = int(predictions[row_index])
            prediction_label = index_to_label[prediction]
            row: dict[str, object] = {
                "sample_index": int(sample_index),
                "prediction_index": prediction,
                "prediction_label": prediction_label,
                "prediction_group": FINE_TO_GROUP.get(prediction_label, ""),
                "confidence": float(probabilities[row_index, prediction]),
            }
            if targets is not None:
                target = int(targets[row_index])
                target_label = index_to_label[target]
                row.update(
                    {
                        "true_index": target,
                        "true_label": target_label,
                        "true_group": FINE_TO_GROUP.get(target_label, ""),
                        "correct": target == prediction,
                    }
                )
            for class_index, fieldname in enumerate(probability_fields):
                row[fieldname] = float(probabilities[row_index, class_index])
            writer.writerow(row)


class PeakRSSMonitor:
    def __init__(self, interval_seconds: float = 0.002):
        self.interval_seconds = interval_seconds
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        try:
            import psutil
        except ImportError:
            return False

        process = psutil.Process(os.getpid())

        def sample() -> None:
            while not self._stop.is_set():
                self.peak_bytes = max(
                    self.peak_bytes,
                    process.memory_info().rss,
                )
                time.sleep(self.interval_seconds)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Full-dataset ONNX Runtime inference and batch-1 benchmark for "
            "Raspberry Pi."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_ASSET_DIR / "student_qkd_int8.onnx",
    )
    parser.add_argument(
        "--left",
        type=Path,
        default=DEFAULT_DATA_DIR / "X_left_test.npy",
    )
    parser.add_argument(
        "--right",
        type=Path,
        default=DEFAULT_DATA_DIR / "X_right_test.npy",
    )
    parser.add_argument(
        "--normalization",
        type=Path,
        default=DEFAULT_ASSET_DIR / "normalization.npz",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_ASSET_DIR / "labels.json",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=DEFAULT_TARGETS_PATH,
        help=(
            "Ground-truth .npy file. y_left_test.npy is detected automatically "
            "when present; otherwise predictions are produced without metrics."
        ),
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Optional single sample index. Omit this option for the full dataset.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--repeat",
        type=int,
        default=100,
        help="Number of batch-1 benchmark runs; dataset inference still runs once.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "onnx_dataset_results.json",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=SCRIPT_DIR / "onnx_dataset_predictions.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required_paths = [
        args.model,
        args.left,
        args.right,
        args.normalization,
        args.labels,
    ]
    if args.targets is not None:
        required_paths.append(args.targets)
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    if args.batch_size <= 0:
        raise ValueError(f"batch-size must be positive, got {args.batch_size}")
    if args.threads <= 0:
        raise ValueError(f"threads must be positive, got {args.threads}")
    if args.warmup < 0:
        raise ValueError(f"warmup cannot be negative, got {args.warmup}")
    if args.repeat <= 0:
        raise ValueError(f"repeat must be positive, got {args.repeat}")
    if args.output.resolve() == args.predictions.resolve():
        raise ValueError("--output and --predictions must be different files.")
    input_paths = {path.resolve() for path in required_paths}
    for option_name, output_path in [
        ("--output", args.output),
        ("--predictions", args.predictions),
    ]:
        if output_path.resolve() in input_paths:
            raise ValueError(
                f"{option_name} cannot overwrite an input file: {output_path}"
            )

    left_all = ensure_channel_first(np.load(args.left))
    right_all = ensure_channel_first(np.load(args.right))
    if left_all.shape != right_all.shape:
        raise ValueError(
            f"Left/right shapes differ: {left_all.shape} vs {right_all.shape}"
        )
    if len(left_all) == 0:
        raise ValueError("The dataset contains no samples.")

    source_sample_count = len(left_all)
    if args.index is None:
        sample_indices = np.arange(source_sample_count, dtype=np.int64)
        left_selected = left_all
        right_selected = right_all
        mode = "full_dataset"
    else:
        if not 0 <= args.index < source_sample_count:
            raise IndexError(
                f"index={args.index}, sample_count={source_sample_count}"
            )
        sample_indices = np.asarray([args.index], dtype=np.int64)
        left_selected = left_all[args.index : args.index + 1]
        right_selected = right_all[args.index : args.index + 1]
        mode = "single_sample"

    left, right = normalize_inputs(
        left_selected,
        right_selected,
        args.normalization,
    )
    index_to_label = load_label_mapping(args.labels)
    label_to_index = {
        label: index for index, label in index_to_label.items()
    }
    targets = (
        load_targets(
            args.targets,
            label_to_index,
            source_sample_count,
            sample_indices,
        )
        if args.targets is not None
        else None
    )

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    rss_monitor = PeakRSSMonitor()
    monitor_active = rss_monitor.start()
    try:
        session = ort.InferenceSession(
            str(args.model),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        benchmark_times_ms = benchmark_single_sample(
            session,
            left,
            right,
            args.warmup,
            args.repeat,
        )
        logits, batch_times_ms = run_dataset_inference(
            session,
            left,
            right,
            args.batch_size,
        )
    finally:
        rss_monitor.stop()

    if logits.shape[1] != len(index_to_label):
        raise ValueError(
            f"Model returned {logits.shape[1]} classes, but labels.json contains "
            f"{len(index_to_label)} labels."
        )

    probabilities = softmax(logits)
    predictions = np.argmax(probabilities, axis=1).astype(np.int64)
    class_names = [
        index_to_label[index] for index in range(len(index_to_label))
    ]
    prediction_counts = {
        label: int(np.count_nonzero(predictions == index))
        for index, label in enumerate(class_names)
    }

    dataset_inference_ms = float(np.sum(batch_times_ms))
    result: dict[str, object] = {
        "model": str(args.model),
        "mode": mode,
        "provider": session.get_providers(),
        "threads": args.threads,
        "source_sample_count": source_sample_count,
        "sample_count": len(sample_indices),
        "batch_size": args.batch_size,
        "class_order": class_names,
        "prediction_count_by_label": prediction_counts,
        "dataset_inference": {
            "total_ms": dataset_inference_ms,
            "batch_count": len(batch_times_ms),
            "batch_latency_ms_mean": float(np.mean(batch_times_ms)),
            "batch_latency_ms_p95": float(np.percentile(batch_times_ms, 95)),
            "estimated_ms_per_sample": dataset_inference_ms / len(sample_indices),
            "throughput_samples_per_second": (
                len(sample_indices) * 1000.0 / dataset_inference_ms
                if dataset_inference_ms > 0
                else None
            ),
        },
        "batch_1_benchmark": {
            "sample_index": int(sample_indices[0]),
            "warmup": args.warmup,
            "repeat": args.repeat,
            "latency_ms_mean": float(np.mean(benchmark_times_ms)),
            "latency_ms_std": float(np.std(benchmark_times_ms)),
            "latency_ms_p50": float(np.percentile(benchmark_times_ms, 50)),
            "latency_ms_p95": float(np.percentile(benchmark_times_ms, 95)),
        },
        "predictions_file": str(args.predictions),
    }
    if monitor_active:
        result["peak_rss_mib"] = rss_monitor.peak_bytes / (1024**2)

    if targets is not None:
        fine_matrix = confusion_matrix(
            targets,
            predictions,
            len(class_names),
        )
        true_groups = fine_to_group_indices(targets, index_to_label)
        predicted_groups = fine_to_group_indices(predictions, index_to_label)
        group_matrix = confusion_matrix(
            true_groups,
            predicted_groups,
            len(GROUP_CLASSES),
        )
        result["evaluation"] = {
            "fine_accuracy": float(np.mean(targets == predictions)),
            "group_accuracy": float(np.mean(true_groups == predicted_groups)),
            "fine_class_order": class_names,
            "group_class_order": GROUP_CLASSES,
            "fine_confusion_matrix": fine_matrix.tolist(),
            "group_confusion_matrix": group_matrix.tolist(),
            "fine_class_report": classification_report(fine_matrix, class_names),
            "group_class_report": classification_report(
                group_matrix,
                GROUP_CLASSES,
            ),
        }

    save_predictions(
        args.predictions,
        sample_indices,
        predictions,
        probabilities,
        index_to_label,
        targets,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
