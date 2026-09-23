import argparse
import csv
import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import psutil
except ImportError:
    psutil = None

from model import Model, make_casual_mask
from model_teacher_original import Model as LegacyModel


DEFAULT_CLASSES = [
    "HC", "H_P", "H_C", "H_F",
    "K_P", "K_F", "K_R",
    "A_F", "A_R", "A_L",
    "C_F", "C_A",
]

# 5-class 그룹 순서는 train.py의 COARSE_CLASSES와 같게 맞춘다.
GROUP_CLASSES = ["HC", "H", "K", "A", "C"]
FINE_TO_GROUP = {
    "HC": "HC",
    "H_P": "H", "H_C": "H", "H_F": "H",
    "A_F": "A", "A_R": "A", "A_L": "A",
    "K_P": "K", "K_F": "K", "K_R": "K",
    "C_F": "C", "C_A": "C",
}


class InferenceDataset(Dataset):
    def __init__(
        self,
        left: np.ndarray,
        right: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> None:
        if len(left) != len(right):
            raise ValueError(
                f"Left/right sample counts differ: {len(left)} != {len(right)}"
            )
        if labels is not None and len(left) != len(labels):
            raise ValueError(
                f"Input/label sample counts differ: {len(left)} != {len(labels)}"
            )

        self.left = torch.from_numpy(left).float()
        self.right = torch.from_numpy(right).float()
        self.labels = None if labels is None else torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.left)

    def __getitem__(self, index: int):
        label = -1 if self.labels is None else self.labels[index]
        return self.left[index], self.right[index], label, index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GRF Transformer inference on Raspberry Pi without matplotlib."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/best_model.pt"),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("processed_data"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("inference_output"),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of PyTorch CPU threads. Raspberry Pi 4 default: 4.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Infer only one test sample. By default, all test samples are used.",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument(
        "--skip-latency",
        action="store_true",
        help="Skip batch-size-1 latency measurement.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(requested)


def ensure_channel_first(x: np.ndarray) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError(f"Input must be 3D [N,C,T] or [N,T,C], got {x.shape}")

    if x.shape[-1] == 5:
        x = np.transpose(x, (0, 2, 1))

    if x.shape[1] != 5:
        raise ValueError(
            f"Expected 5 sensor channels after conversion, got shape {x.shape}"
        )

    return np.ascontiguousarray(x, dtype=np.float32)


def calculate_normalization_stats(
    data_dir: Path,
    normalization_path: Path,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left_train_path = data_dir / "X_left_train.npy"
    right_train_path = data_dir / "X_right_train.npy"

    if not left_train_path.exists() or not right_train_path.exists():
        raise FileNotFoundError(
            "normalization.npz does not exist, and training arrays required to "
            "reconstruct the original normalization are missing. Expected:\n"
            f"  {left_train_path}\n"
            f"  {right_train_path}"
        )

    left_train = ensure_channel_first(
        np.load(left_train_path).astype(np.float32)
    )
    right_train = ensure_channel_first(
        np.load(right_train_path).astype(np.float32)
    )

    left_mean = left_train.mean(axis=(0, 2), keepdims=True)
    left_std = left_train.std(axis=(0, 2), keepdims=True)
    right_mean = right_train.mean(axis=(0, 2), keepdims=True)
    right_std = right_train.std(axis=(0, 2), keepdims=True)

    np.savez(
        normalization_path,
        left_mean=left_mean,
        left_std=left_std,
        right_mean=right_mean,
        right_std=right_std,
        eps=np.array(eps, dtype=np.float32),
    )
    print(f"Created normalization file: {normalization_path}")

    return left_mean, left_std, right_mean, right_std


def load_normalization_stats(
    data_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    normalization_path = data_dir / "normalization.npz"

    if not normalization_path.exists():
        left_mean, left_std, right_mean, right_std = calculate_normalization_stats(
            data_dir,
            normalization_path,
        )
        return left_mean, left_std, right_mean, right_std, 1e-6

    with np.load(normalization_path) as stats:
        required = {"left_mean", "left_std", "right_mean", "right_std"}
        missing = required.difference(stats.files)
        if missing:
            raise KeyError(
                f"{normalization_path} is missing keys: {sorted(missing)}"
            )

        eps = float(stats["eps"]) if "eps" in stats.files else 1e-6
        return (
            stats["left_mean"].astype(np.float32),
            stats["left_std"].astype(np.float32),
            stats["right_mean"].astype(np.float32),
            stats["right_std"].astype(np.float32),
            eps,
        )


def normalize_test_data(
    left: np.ndarray,
    right: np.ndarray,
    data_dir: Path,
    normalization: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if normalization is None:
        left_mean, left_std, right_mean, right_std, eps = load_normalization_stats(
            data_dir
        )
    else:
        required = {"left_mean", "left_std", "right_mean", "right_std"}
        missing = required - set(normalization)
        if missing:
            raise KeyError(
                f"checkpoint normalization is missing keys: {sorted(missing)}"
            )
        left_mean = np.asarray(normalization["left_mean"], dtype=np.float32)
        left_std = np.asarray(normalization["left_std"], dtype=np.float32)
        right_mean = np.asarray(normalization["right_mean"], dtype=np.float32)
        right_std = np.asarray(normalization["right_std"], dtype=np.float32)
        eps = float(normalization.get("eps", 1e-6))

    left = (left - left_mean) / (left_std + eps)
    right = (right - right_mean) / (right_std + eps)

    return (
        np.ascontiguousarray(left, dtype=np.float32),
        np.ascontiguousarray(right, dtype=np.float32),
    )


def load_test_data(
    data_dir: Path,
    label_to_idx: Dict[str, int],
    normalization: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    left_path = data_dir / "X_left_test.npy"
    right_path = data_dir / "X_right_test.npy"
    label_path = data_dir / "y_left_test.npy"

    left = ensure_channel_first(np.load(left_path).astype(np.float32))
    right = ensure_channel_first(np.load(right_path).astype(np.float32))
    left, right = normalize_test_data(
        left, right, data_dir, normalization=normalization
    )

    labels: Optional[np.ndarray] = None
    if label_path.exists():
        raw_labels = np.load(label_path, allow_pickle=True)
        unknown = sorted({str(label) for label in raw_labels} - set(label_to_idx))
        if unknown:
            raise KeyError(f"Unknown labels in {label_path}: {unknown}")
        labels = np.asarray(
            [label_to_idx[str(label)] for label in raw_labels],
            dtype=np.int64,
        )

    return left, right, labels


def load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # weights_only=False is needed for full training checkpoint dictionaries on
    # recent PyTorch versions. Only load checkpoints you trust.
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dictionary or a state_dict.")

    return checkpoint


def extract_state_dict(checkpoint: dict) -> dict:
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]

    # Also accept a raw state_dict.
    if checkpoint and all(isinstance(key, str) for key in checkpoint):
        return checkpoint

    raise KeyError("Checkpoint does not contain 'model_state_dict'.")


def normalize_label_mapping(checkpoint: dict) -> Tuple[Dict[str, int], Dict[int, str]]:
    raw_mapping = checkpoint.get("label_to_idx")

    if raw_mapping is None:
        label_to_idx = {label: index for index, label in enumerate(DEFAULT_CLASSES)}
    else:
        label_to_idx = {str(label): int(index) for label, index in raw_mapping.items()}

    idx_to_label = {index: label for label, index in label_to_idx.items()}
    expected_indices = set(range(len(label_to_idx)))
    if set(idx_to_label) != expected_indices:
        raise ValueError(
            "Class indices in label_to_idx must be contiguous and start from zero."
        )

    return label_to_idx, idx_to_label


def infer_config(
    checkpoint: dict,
    state_dict: dict,
    sensor_dim: int,
    max_len: int,
) -> dict:
    config = dict(checkpoint.get("config", {}))

    # Recover dimensions from tensor shapes when possible.
    config.setdefault("embed_dim", int(state_dict["classifier.weight"].shape[1]))
    config.setdefault("sensor_dim", sensor_dim)
    config.setdefault("ff_dim", int(state_dict["encoder.0.ff.0.weight"].shape[0]))
    config.setdefault("max_len", max_len)
    config.setdefault("dropout", 0.1)
    config.setdefault("num_heads", 8)

    encoder_indices = {
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("encoder.") and key.split(".")[1].isdigit()
    }
    config.setdefault("num_layers", len(encoder_indices))

    if int(config["sensor_dim"]) != sensor_dim:
        raise ValueError(
            f"Checkpoint sensor_dim={config['sensor_dim']} but input has {sensor_dim} channels."
        )
    if int(config["max_len"]) < max_len:
        raise ValueError(
            f"Checkpoint max_len={config['max_len']} is shorter than input length {max_len}."
        )

    return config


def build_model(
    checkpoint: dict,
    state_dict: dict,
    num_classes: int,
    sensor_dim: int,
    max_len: int,
    device: torch.device,
) -> Tuple[Model, dict]:
    config = infer_config(checkpoint, state_dict, sensor_dim, max_len)
    architecture_version = int(
        config.get(
            "architecture_version",
            1 if any(key.startswith("decoder.") for key in state_dict) else 2,
        )
    )
    model_kwargs = {
        "num_classes": num_classes,
        "embed_dim": int(config["embed_dim"]),
        "sensor_dim": int(config["sensor_dim"]),
        "num_heads": int(config["num_heads"]),
        "num_layers": int(config["num_layers"]),
        "ff_dim": int(config["ff_dim"]),
        "dropout": float(config["dropout"]),
        "max_len": int(config["max_len"]),
    }
    if architecture_version >= 2:
        model_kwargs.update(
            {
                "num_groups": int(config.get("num_groups", 5)),
                "temporal_kernel_size": int(
                    config.get("temporal_kernel_size", 5)
                ),
            }
        )
        model = Model(**model_kwargs).to(device)
    else:
        model = LegacyModel(**model_kwargs).to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model, config


@torch.inference_mode()
def run_inference(
    model: Model,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_indices = []
    all_labels = []
    all_predictions = []
    all_probabilities = []

    for left, right, labels, indices in loader:
        left = left.to(device, non_blocking=True)
        right = right.to(device, non_blocking=True)

        tgt_seq = torch.zeros(
            (left.size(0), 1),
            dtype=torch.long,
            device=device,
        )
        tgt_mask = make_casual_mask(1, device)

        sequence_logits = model(left, right, tgt_seq, tgt_mask=tgt_mask)
        logits = sequence_logits[-1]
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

        all_indices.append(indices.numpy())
        all_labels.append(labels.numpy())
        all_predictions.append(predictions.cpu().numpy())
        all_probabilities.append(probabilities.cpu().numpy())

    return (
        np.concatenate(all_indices),
        np.concatenate(all_labels),
        np.concatenate(all_predictions),
        np.concatenate(all_probabilities),
    )


def make_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_index, pred_index in zip(y_true, y_pred):
        matrix[int(true_index), int(pred_index)] += 1
    return matrix


def calculate_report(matrix: np.ndarray, class_names: list[str]) -> list[dict]:
    rows = []
    for index, class_name in enumerate(class_names):
        true_positive = int(matrix[index, index])
        false_positive = int(matrix[:, index].sum() - true_positive)
        false_negative = int(matrix[index, :].sum() - true_positive)
        support = int(matrix[index, :].sum())

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive > 0
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative > 0
            else 0.0
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        rows.append(
            {
                "class": class_name,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": support,
            }
        )
    return rows


def save_report(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["class", "precision", "recall", "f1_score", "support"],
        )
        writer.writeheader()
        writer.writerows(rows)



def fine_indices_to_group_indices(
    indices: np.ndarray,
    idx_to_label: Dict[int, str],
) -> np.ndarray:
    group_to_idx = {label: index for index, label in enumerate(GROUP_CLASSES)}
    converted = []

    for class_index in indices:
        fine_label = idx_to_label[int(class_index)]
        if fine_label not in FINE_TO_GROUP:
            raise KeyError(f"No group mapping defined for fine label: {fine_label}")
        converted.append(group_to_idx[FINE_TO_GROUP[fine_label]])

    return np.asarray(converted, dtype=np.int64)


def save_predictions(
    path: Path,
    sample_indices: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    idx_to_label: Dict[int, str],
) -> None:
    class_names = [idx_to_label[index] for index in range(len(idx_to_label))]
    has_labels = np.all(labels >= 0)

    fieldnames = ["sample_index"]
    if has_labels:
        fieldnames.extend(["true_index", "true_label", "true_group"])
    fieldnames.extend(
        ["prediction_index", "prediction_label", "prediction_group", "confidence"]
    )
    if has_labels:
        fieldnames.append("correct")
    fieldnames.extend([f"prob_{name}" for name in class_names])

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row_index, sample_index in enumerate(sample_indices):
            pred_index = int(predictions[row_index])
            pred_label = idx_to_label[pred_index]
            row = {
                "sample_index": int(sample_index),
                "prediction_index": pred_index,
                "prediction_label": pred_label,
                "prediction_group": FINE_TO_GROUP.get(pred_label, ""),
                "confidence": float(probabilities[row_index, pred_index]),
            }

            if has_labels:
                true_index = int(labels[row_index])
                true_label = idx_to_label[true_index]
                row.update(
                    {
                        "true_index": true_index,
                        "true_label": true_label,
                        "true_group": FINE_TO_GROUP.get(true_label, ""),
                        "correct": bool(true_index == pred_index),
                    }
                )

            for class_index, class_name in enumerate(class_names):
                row[f"prob_{class_name}"] = float(probabilities[row_index, class_index])

            writer.writerow(row)


@torch.inference_mode()
def measure_latency(
    model: Model,
    left: np.ndarray,
    right: np.ndarray,
    device: torch.device,
    warmup: int,
    repeat: int,
) -> dict:
    left_tensor = torch.from_numpy(left[:1]).to(device)
    right_tensor = torch.from_numpy(right[:1]).to(device)
    tgt_seq = torch.zeros((1, 1), dtype=torch.long, device=device)
    tgt_mask = make_casual_mask(1, device)

    for _ in range(warmup):
        model(left_tensor, right_tensor, tgt_seq, tgt_mask=tgt_mask)
    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed_ms = []
    for _ in range(repeat):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        model(left_tensor, right_tensor, tgt_seq, tgt_mask=tgt_mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms.append((time.perf_counter() - start) * 1000.0)

    return {
        "mean_ms": float(np.mean(elapsed_ms)),
        "std_ms": float(np.std(elapsed_ms)),
        "p50_ms": float(np.percentile(elapsed_ms, 50)),
        "p95_ms": float(np.percentile(elapsed_ms, 95)),
        "warmup": warmup,
        "repeat": repeat,
        "batch_size": 1,
        "device": str(device),
    }


def main() -> None:
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.threads < 1:
        raise ValueError("--threads must be at least 1")

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # This can only be configured before parallel work begins.
        pass

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Device: {device}")

    checkpoint = load_checkpoint(args.checkpoint, device)
    state_dict = extract_state_dict(checkpoint)
    label_to_idx, idx_to_label = normalize_label_mapping(checkpoint)

    checkpoint_normalization = checkpoint.get("normalization")
    if checkpoint_normalization is None:
        checkpoint_normalization = checkpoint.get("config", {}).get("normalization")
    left, right, labels = load_test_data(
        args.data_dir,
        label_to_idx,
        normalization=checkpoint_normalization,
    )

    original_indices = np.arange(len(left), dtype=np.int64)
    if args.index is not None:
        if args.index < 0 or args.index >= len(left):
            raise IndexError(
                f"--index must be between 0 and {len(left) - 1}, got {args.index}"
            )
        selected = slice(args.index, args.index + 1)
        left = left[selected]
        right = right[selected]
        original_indices = original_indices[selected]
        if labels is not None:
            labels = labels[selected]

    model, config = build_model(
        checkpoint=checkpoint,
        state_dict=state_dict,
        num_classes=len(label_to_idx),
        sensor_dim=left.shape[1],
        max_len=left.shape[2],
        device=device,
    )

    dataset = InferenceDataset(left, right, labels)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # Measure only this Python process. This is not total system RAM.
    if psutil is None:
        raise RuntimeError(
            "psutil is required for process RSS measurement. "
            "Install it with: python -m pip install psutil"
        )

    process = psutil.Process(os.getpid())
    mib = 1024.0 * 1024.0

    rss_before_bytes = int(process.memory_info().rss)
    peak_rss_bytes = [rss_before_bytes]
    monitor_stop = threading.Event()

    def monitor_process_rss() -> None:
        while not monitor_stop.wait(0.001):
            try:
                current_rss = int(process.memory_info().rss)
                if current_rss > peak_rss_bytes[0]:
                    peak_rss_bytes[0] = current_rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        cuda_allocated_before = int(torch.cuda.memory_allocated(device))
        cuda_reserved_before = int(torch.cuda.memory_reserved(device))
    else:
        cuda_allocated_before = 0
        cuda_reserved_before = 0

    monitor_thread = threading.Thread(
        target=monitor_process_rss,
        daemon=True,
    )
    monitor_thread.start()

    try:
        local_indices, y_true, y_pred, probabilities = run_inference(
            model,
            loader,
            device,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    finally:
        monitor_stop.set()
        monitor_thread.join()

    rss_after_bytes = int(process.memory_info().rss)
    peak_rss_bytes[0] = max(
        peak_rss_bytes[0],
        rss_before_bytes,
        rss_after_bytes,
    )

    runtime_memory = {
        "process_id": os.getpid(),
        "scope": "current Python process only",
        "rss_before_mib": rss_before_bytes / mib,
        "rss_after_mib": rss_after_bytes / mib,
        "peak_rss_mib": peak_rss_bytes[0] / mib,
        "peak_rss_increase_mib": max(
            0,
            peak_rss_bytes[0] - rss_before_bytes,
        ) / mib,
        "sampling_interval_ms": 1.0,
        "source": "psutil.Process(os.getpid()).memory_info().rss",
        "includes_system_memory": False,
        "includes_dataloader_workers": False,
    }

    if device.type == "cuda":
        runtime_memory["cuda_vram"] = {
            "allocated_before_mib": cuda_allocated_before / mib,
            "reserved_before_mib": cuda_reserved_before / mib,
            "peak_allocated_mib": (
                torch.cuda.max_memory_allocated(device) / mib
            ),
            "peak_reserved_mib": (
                torch.cuda.max_memory_reserved(device) / mib
            ),
        }
    else:
        runtime_memory["cuda_vram"] = None

    sample_indices = original_indices[local_indices]

    predictions_path = args.output_dir / "predictions.csv"
    save_predictions(
        predictions_path,
        sample_indices,
        y_true,
        y_pred,
        probabilities,
        idx_to_label,
    )

    summary = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "num_samples": len(dataset),
        "batch_size": args.batch_size,
        "threads": args.threads,
        "model_config": config,
        "predictions": str(predictions_path),
        "runtime_memory": runtime_memory,
    }

    print(
        "Peak process RSS: "
        f"{runtime_memory['peak_rss_mib']:.3f} MiB "
        f"(increase: {runtime_memory['peak_rss_increase_mib']:.3f} MiB)"
    )
    if runtime_memory["cuda_vram"] is not None:
        print(
            "Peak CUDA VRAM: "
            f"allocated "
            f"{runtime_memory['cuda_vram']['peak_allocated_mib']:.3f} MiB, "
            f"reserved "
            f"{runtime_memory['cuda_vram']['peak_reserved_mib']:.3f} MiB"
        )

    if np.all(y_true >= 0):
        fine_class_names = [
            idx_to_label[index] for index in range(len(idx_to_label))
        ]
        fine_accuracy = float(np.mean(y_true == y_pred))
        fine_matrix = make_confusion_matrix(
            y_true,
            y_pred,
            len(fine_class_names),
        )

        np.save(args.output_dir / "confusion_matrix.npy", fine_matrix)
        save_report(
            args.output_dir / "classification_report.csv",
            calculate_report(fine_matrix, fine_class_names),
        )

        y_true_group = fine_indices_to_group_indices(y_true, idx_to_label)
        y_pred_group = fine_indices_to_group_indices(y_pred, idx_to_label)
        group_accuracy = float(np.mean(y_true_group == y_pred_group))
        group_matrix = make_confusion_matrix(
            y_true_group,
            y_pred_group,
            len(GROUP_CLASSES),
        )

        np.save(args.output_dir / "grouped_confusion_matrix.npy", group_matrix)
        save_report(
            args.output_dir / "grouped_classification_report.csv",
            calculate_report(group_matrix, GROUP_CLASSES),
        )

        summary["fine_accuracy"] = fine_accuracy
        summary["group_accuracy"] = group_accuracy
        print(f"Fine-class accuracy: {fine_accuracy:.4f}")
        print(f"Grouped accuracy:    {group_accuracy:.4f}")

    if not args.skip_latency:
        summary["latency"] = measure_latency(
            model,
            left,
            right,
            device,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        print(
            "Latency: "
            f"{summary['latency']['mean_ms']:.3f} ± "
            f"{summary['latency']['std_ms']:.3f} ms"
        )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Predictions saved to: {predictions_path}")
    print(f"Summary saved to:     {summary_path}")


if __name__ == "__main__":
    main()
