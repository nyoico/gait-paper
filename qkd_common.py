import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from plot_style import (
    CONFUSION_CELL_FONTSIZE,
    CONFUSION_CMAP,
    FIGURE_DPI,
    apply_paper_style,
    confusion_text_color,
)

# 이 모듈을 쓰는 모든 스크립트의 그림에 논문용 폰트/해상도 설정을 적용합니다.
apply_paper_style()


VALID_CLASSES = [
    "HC", "H_P", "H_C", "H_F",
    "K_P", "K_F", "K_R",
    "A_F", "A_R", "A_L",
    "C_F", "C_A",
]
# 5-class 그룹 순서는 train.py의 COARSE_CLASSES와 같게 맞춘다.
GROUP_CLASSES = ["HC", "H", "K", "A", "C"]

LABEL_TO_IDX = {label: idx for idx, label in enumerate(VALID_CLASSES)}
IDX_TO_LABEL = {idx: label for label, idx in LABEL_TO_IDX.items()}
GROUP_LABEL_TO_IDX = {label: idx for idx, label in enumerate(GROUP_CLASSES)}

FINE_TO_GROUP = {
    "HC": "HC",
    "H_P": "H", "H_C": "H", "H_F": "H",
    "A_F": "A", "A_R": "A", "A_L": "A",
    "K_P": "K", "K_F": "K", "K_R": "K",
    "C_F": "C", "C_A": "C",
}


class GRFDataset(Dataset):
    def __init__(self, left: np.ndarray, right: np.ndarray, labels: np.ndarray):
        if not (len(left) == len(right) == len(labels)):
            raise ValueError(
                f"Length mismatch: left={len(left)}, right={len(right)}, labels={len(labels)}"
            )
        self.left = torch.from_numpy(left).float()
        self.right = torch.from_numpy(right).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.left[idx], self.right[idx], self.labels[idx]



def load_checkpoint(path: Path, map_location=None) -> Dict[str, Any]:
    """Load a trusted local training checkpoint across PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_channel_first(x: np.ndarray) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError(f"Input must be 3-D, got shape={x.shape}")
    if x.shape[-1] == 5:
        x = np.transpose(x, (0, 2, 1))
    if x.shape[1] != 5:
        raise ValueError(f"Expected five sensor channels, got shape={x.shape}")
    return x.astype(np.float32)


def encode_labels(labels: Sequence[Any]) -> np.ndarray:
    encoded: List[int] = []
    for label in labels:
        key = str(label)
        if key not in LABEL_TO_IDX:
            raise ValueError(
                f"Unknown label {key!r}. Expected one of {VALID_CLASSES}."
            )
        encoded.append(LABEL_TO_IDX[key])
    return np.asarray(encoded, dtype=np.int64)


def _normalization_stats(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=(0, 2), keepdims=True)
    std = x.std(axis=(0, 2), keepdims=True)
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_normalization(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    return ((x - mean) / (std + eps)).astype(np.float32)


def load_processed_arrays(processed_dir: Path) -> Dict[str, np.ndarray]:
    required = [
        "X_left_train.npy", "X_left_test.npy",
        "X_right_train.npy", "X_right_test.npy",
        "y_left_train.npy", "y_left_test.npy",
        "y_right_train.npy", "y_right_test.npy",
    ]
    missing = [name for name in required if not (processed_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing files in {processed_dir}: {', '.join(missing)}"
        )

    arrays: Dict[str, np.ndarray] = {}
    for name in required:
        allow_pickle = name.startswith("y_")
        arrays[name.removesuffix(".npy")] = np.load(
            processed_dir / name,
            allow_pickle=allow_pickle,
        )
    return arrays


def prepare_data(
    processed_dir: Path,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    seed: int,
    split_indices: Optional[Dict[str, List[int]]] = None,
    normalization: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    arrays = load_processed_arrays(processed_dir)

    left_train = ensure_channel_first(arrays["X_left_train"])
    left_test = ensure_channel_first(arrays["X_left_test"])
    right_train = ensure_channel_first(arrays["X_right_train"])
    right_test = ensure_channel_first(arrays["X_right_test"])

    raw_left_train_labels = np.asarray(arrays["y_left_train"]).astype(str)
    raw_right_train_labels = np.asarray(arrays["y_right_train"]).astype(str)
    raw_left_test_labels = np.asarray(arrays["y_left_test"]).astype(str)
    raw_right_test_labels = np.asarray(arrays["y_right_test"]).astype(str)

    if not np.array_equal(raw_left_train_labels, raw_right_train_labels):
        mismatch = np.flatnonzero(raw_left_train_labels != raw_right_train_labels)
        raise ValueError(
            f"Left/right train labels differ at {len(mismatch)} samples. "
            f"First mismatches: {mismatch[:10].tolist()}"
        )
    if not np.array_equal(raw_left_test_labels, raw_right_test_labels):
        mismatch = np.flatnonzero(raw_left_test_labels != raw_right_test_labels)
        raise ValueError(
            f"Left/right test labels differ at {len(mismatch)} samples. "
            f"First mismatches: {mismatch[:10].tolist()}"
        )

    y_train = encode_labels(raw_left_train_labels)
    y_test = encode_labels(raw_left_test_labels)

    if normalization is None:
        left_mean, left_std = _normalization_stats(left_train)
        right_mean, right_std = _normalization_stats(right_train)
        normalization = {
            "left_mean": left_mean,
            "left_std": left_std,
            "right_mean": right_mean,
            "right_std": right_std,
        }
    else:
        normalization = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in normalization.items()
        }

    left_train = _apply_normalization(
        left_train,
        normalization["left_mean"],
        normalization["left_std"],
    )
    left_test = _apply_normalization(
        left_test,
        normalization["left_mean"],
        normalization["left_std"],
    )
    right_train = _apply_normalization(
        right_train,
        normalization["right_mean"],
        normalization["right_std"],
    )
    right_test = _apply_normalization(
        right_test,
        normalization["right_mean"],
        normalization["right_std"],
    )

    full_train_dataset = GRFDataset(left_train, right_train, y_train)
    test_dataset = GRFDataset(left_test, right_test, y_test)

    if split_indices is None:
        val_size = max(1, int(len(full_train_dataset) * val_ratio))
        train_size = len(full_train_dataset) - val_size
        permutation = torch.randperm(
            len(full_train_dataset),
            generator=torch.Generator().manual_seed(seed),
        ).tolist()
        train_indices = permutation[:train_size]
        val_indices = permutation[train_size:]
        split_indices = {
            "train": train_indices,
            "val": val_indices,
        }
    else:
        train_indices = [int(i) for i in split_indices["train"]]
        val_indices = [int(i) for i in split_indices["val"]]

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "sensor_dim": left_train.shape[1],
        "max_len": left_train.shape[2],
        "normalization": normalization,
        "split_indices": split_indices,
    }


def make_tgt_inputs(batch_size: int, device: torch.device):
    tgt_seq = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
    tgt_mask = torch.zeros((1, 1), dtype=torch.bool, device=device)
    return tgt_seq, tgt_mask


def extract_last_logits(output: torch.Tensor) -> torch.Tensor:
    if output.ndim == 2:
        return output
    if output.ndim == 3:
        return output[-1]
    raise ValueError(f"Unexpected model output shape: {tuple(output.shape)}")


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_true: List[int] = []
    all_pred: List[int] = []

    for left, right, labels in loader:
        left = left.to(device, non_blocking=True)
        right = right.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        tgt_seq, tgt_mask = make_tgt_inputs(labels.size(0), device)

        logits = extract_last_logits(
            model(left, right, tgt_seq, tgt_mask=tgt_mask)
        )
        loss = torch.nn.functional.cross_entropy(logits, labels)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)
        all_true.extend(labels.cpu().tolist())
        all_pred.extend(preds.cpu().tolist())

    y_true = np.asarray(all_true, dtype=np.int64)
    y_pred = np.asarray(all_pred, dtype=np.int64)
    fine_accuracy = total_correct / max(total_count, 1)

    group_true = fine_to_group_indices(y_true)
    group_pred = fine_to_group_indices(y_pred)
    group_accuracy = (
        float((group_true == group_pred).mean()) if len(group_true) else 0.0
    )

    return {
        "loss": total_loss / max(total_count, 1),
        # Keep `accuracy` for backward compatibility. It is the 12-class accuracy.
        "accuracy": fine_accuracy,
        "fine_accuracy": fine_accuracy,
        "group_accuracy": group_accuracy,
        "accuracy_12class": fine_accuracy,
        "accuracy_5class": group_accuracy,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
    }


def fine_to_group_indices(values: np.ndarray) -> np.ndarray:
    result = []
    for value in values:
        fine_name = IDX_TO_LABEL[int(value)]
        group_name = FINE_TO_GROUP[fine_name]
        result.append(GROUP_LABEL_TO_IDX[group_name])
    return np.asarray(result, dtype=np.int64)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    matrix = np.zeros((n, n), dtype=np.int64)
    for true_value, pred_value in zip(y_true, y_pred):
        matrix[int(true_value), int(pred_value)] += 1
    return matrix


def classification_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
) -> List[Dict[str, Any]]:
    matrix = confusion_matrix(y_true, y_pred, len(class_names))
    rows: List[Dict[str, Any]] = []
    for idx, name in enumerate(class_names):
        tp = int(matrix[idx, idx])
        fp = int(matrix[:, idx].sum() - tp)
        fn = int(matrix[idx, :].sum() - tp)
        support = int(matrix[idx, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "class": name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
    return rows


def save_classification_csv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    path: Path,
) -> None:
    rows = classification_rows(y_true, y_pred, class_names)
    lines = ["class,precision,recall,f1-score,support"]
    for row in rows:
        lines.append(
            f"{row['class']},{row['precision']:.6f},{row['recall']:.6f},"
            f"{row['f1']:.6f},{row['support']}"
        )
    accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    lines.append(f"accuracy,{accuracy:.6f}")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_confusion(
    matrix: np.ndarray,
    labels: Sequence[str],
    title: str,
    path: Path,
) -> None:
    plt.figure(figsize=(12, 10))
    plt.imshow(matrix, cmap=CONFUSION_CMAP)
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(np.arange(len(labels)), labels)

    # 색이 진한 셀에서는 흰 글씨로 바꿔 숫자가 항상 읽히게 합니다.
    vmax = float(matrix.max()) if matrix.size else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=CONFUSION_CELL_FONTSIZE,
                color=confusion_text_color(matrix[i, j], vmax),
            )
    plt.tight_layout()
    plt.savefig(path, dpi=FIGURE_DPI)
    plt.close()


def save_evaluation_outputs(
    result: Dict[str, Any],
    output_dir: Path,
    prefix: str,
) -> Dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true = result["y_true"]
    y_pred = result["y_pred"]

    fine_cm = confusion_matrix(y_true, y_pred, len(VALID_CLASSES))
    np.save(output_dir / f"{prefix}_fine_confusion_matrix.npy", fine_cm)
    plot_confusion(
        fine_cm,
        VALID_CLASSES,
        f"{prefix} Fine-class Confusion Matrix",
        output_dir / f"{prefix}_fine_confusion_matrix.png",
    )
    save_classification_csv(
        y_true,
        y_pred,
        VALID_CLASSES,
        output_dir / f"{prefix}_fine_classification_report.csv",
    )

    group_true = fine_to_group_indices(y_true)
    group_pred = fine_to_group_indices(y_pred)
    group_cm = confusion_matrix(group_true, group_pred, len(GROUP_CLASSES))
    np.save(output_dir / f"{prefix}_group_confusion_matrix.npy", group_cm)
    plot_confusion(
        group_cm,
        GROUP_CLASSES,
        f"{prefix} Grouped Confusion Matrix",
        output_dir / f"{prefix}_group_confusion_matrix.png",
    )
    save_classification_csv(
        group_true,
        group_pred,
        GROUP_CLASSES,
        output_dir / f"{prefix}_group_classification_report.csv",
    )

    fine_accuracy = float(result.get("fine_accuracy", result["accuracy"]))
    group_accuracy = float((group_true == group_pred).mean()) if len(group_true) else 0.0
    metrics = {
        "fine_loss": float(result["loss"]),
        "fine_accuracy": fine_accuracy,
        "group_accuracy": group_accuracy,
        # Explicit aliases make the paper-facing class cardinality unambiguous.
        "accuracy_12class": fine_accuracy,
        "accuracy_5class": group_accuracy,
    }
    (output_dir / f"{prefix}_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    return metrics


@torch.no_grad()
def measure_latency(
    model: torch.nn.Module,
    dataset: Dataset,
    device: torch.device,
    warmup: int = 20,
    repeat: int = 100,
) -> Dict[str, Any]:
    model.eval()
    left, right, _ = dataset[0]
    left = left.unsqueeze(0).to(device)
    right = right.unsqueeze(0).to(device)
    tgt_seq, tgt_mask = make_tgt_inputs(1, device)

    for _ in range(warmup):
        model(left, right, tgt_seq, tgt_mask=tgt_mask)
        if device.type == "cuda":
            torch.cuda.synchronize()

    timings = []
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        for _ in range(repeat):
            start_event.record()
            model(left, right, tgt_seq, tgt_mask=tgt_mask)
            end_event.record()
            torch.cuda.synchronize()
            timings.append(start_event.elapsed_time(end_event))
    else:
        for _ in range(repeat):
            start = time.perf_counter()
            model(left, right, tgt_seq, tgt_mask=tgt_mask)
            timings.append((time.perf_counter() - start) * 1000.0)

    return {
        "mean_ms": float(np.mean(timings)),
        "std_ms": float(np.std(timings)),
        "warmup": warmup,
        "repeat": repeat,
        "device": str(device),
        "batch_size": 1,
        "note": "Fake-quantized PyTorch latency; not real INT8-kernel latency.",
    }
