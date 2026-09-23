from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


# ============================================================
# 직접 경로 / 실행 설정 지정
# ============================================================
MODEL_PATH = Path("deploy/student_qkd_int8.onnx")
PROCESSED_DIR = Path("processed_data")
NORMALIZATION_PATH = Path("deploy/normalization.npz")
LABELS_PATH = Path("deploy/labels.json")

BATCH_SIZE = 64
THREADS = 4

# 결과 JSON을 저장하지 않으려면 None으로 설정
OUTPUT_PATH: Path | None = Path("output/onnx_test_result.json")


FINE_TO_GROUP = {
    "HC": "HC",
    "H_P": "H", "H_C": "H", "H_F": "H",
    "K_P": "K", "K_F": "K", "K_R": "K",
    "A_F": "A", "A_R": "A", "A_L": "A",
    "C_F": "C", "C_A": "C",
}

GROUP_ORDER = ["HC", "H", "K", "A", "C"]


def ensure_channel_first(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)

    if x.ndim != 3:
        raise ValueError(f"Expected 3-D input, got {x.shape}")

    # [N, T, 5] -> [N, 5, T]
    if x.shape[-1] == 5:
        x = np.transpose(x, (0, 2, 1))

    if x.shape[1] != 5:
        raise ValueError(f"Expected [N, 5, T], got {x.shape}")

    return np.ascontiguousarray(x, dtype=np.float32)


def normalize(
    left: np.ndarray,
    right: np.ndarray,
    normalization_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    stats = np.load(normalization_path)

    left = (left - stats["left_mean"]) / (stats["left_std"] + 1e-6)
    right = (right - stats["right_mean"]) / (stats["right_std"] + 1e-6)

    return (
        np.ascontiguousarray(left, dtype=np.float32),
        np.ascontiguousarray(right, dtype=np.float32),
    )


def load_label_mapping(path: Path) -> tuple[dict[int, str], dict[str, int]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    idx_to_label = {int(index): label for index, label in raw.items()}
    label_to_idx = {label: index for index, label in idx_to_label.items()}
    return idx_to_label, label_to_idx


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def main() -> None:
    # --------------------------------------------------------
    # 파일 확인
    # --------------------------------------------------------
    required_paths = [
        MODEL_PATH,
        NORMALIZATION_PATH,
        LABELS_PATH,
        PROCESSED_DIR / "X_left_test.npy",
        PROCESSED_DIR / "X_right_test.npy",
        PROCESSED_DIR / "y_left_test.npy",
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    # --------------------------------------------------------
    # Test data load
    # --------------------------------------------------------
    left = ensure_channel_first(
        np.load(PROCESSED_DIR / "X_left_test.npy")
    )
    right = ensure_channel_first(
        np.load(PROCESSED_DIR / "X_right_test.npy")
    )

    if len(left) != len(right):
        raise ValueError(
            f"Left/right sample counts differ: {len(left)} vs {len(right)}"
        )

    left, right = normalize(
        left,
        right,
        NORMALIZATION_PATH,
    )

    # --------------------------------------------------------
    # Label load
    # --------------------------------------------------------
    y_left_raw = np.load(
        PROCESSED_DIR / "y_left_test.npy",
        allow_pickle=True,
    )

    y_right_path = PROCESSED_DIR / "y_right_test.npy"
    if y_right_path.exists():
        y_right_raw = np.load(y_right_path, allow_pickle=True)
        if not np.array_equal(
            y_left_raw.astype(str),
            y_right_raw.astype(str),
        ):
            raise ValueError("Left/right test labels do not match.")

    idx_to_label, label_to_idx = load_label_mapping(LABELS_PATH)

    y_true = np.asarray(
        [label_to_idx[str(label)] for label in y_left_raw],
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # ONNX Runtime session
    # --------------------------------------------------------
    options = ort.SessionOptions()
    options.intra_op_num_threads = THREADS
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    # --------------------------------------------------------
    # Full test-set inference
    # --------------------------------------------------------
    predictions: list[np.ndarray] = []

    for start in range(0, len(left), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(left))

        logits = session.run(
            ["logits"],
            {
                "sensor_left": left[start:end],
                "sensor_right": right[start:end],
            },
        )[0]

        predictions.append(
            np.argmax(logits, axis=1).astype(np.int64)
        )

    y_pred = np.concatenate(predictions)

    # --------------------------------------------------------
    # 12-class accuracy
    # --------------------------------------------------------
    accuracy_12class = float(np.mean(y_true == y_pred))

    # --------------------------------------------------------
    # 5-class grouped accuracy
    # HC / H / K / A / C
    # --------------------------------------------------------
    group_to_idx = {
        label: index
        for index, label in enumerate(GROUP_ORDER)
    }

    y_true_group = np.asarray(
        [
            group_to_idx[FINE_TO_GROUP[idx_to_label[int(index)]]]
            for index in y_true
        ],
        dtype=np.int64,
    )

    y_pred_group = np.asarray(
        [
            group_to_idx[FINE_TO_GROUP[idx_to_label[int(index)]]]
            for index in y_pred
        ],
        dtype=np.int64,
    )

    accuracy_5class = float(
        np.mean(y_true_group == y_pred_group)
    )

    # --------------------------------------------------------
    # Confusion matrices
    # --------------------------------------------------------
    cm_12class = confusion_matrix(
        y_true,
        y_pred,
        len(idx_to_label),
    )

    cm_5class = confusion_matrix(
        y_true_group,
        y_pred_group,
        len(GROUP_ORDER),
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------
    result = {
        "model": str(MODEL_PATH),
        "sample_count": int(len(y_true)),
        "accuracy_12class": accuracy_12class,
        "accuracy_5class": accuracy_5class,
        "class_order_12class": [
            idx_to_label[index]
            for index in range(len(idx_to_label))
        ],
        "class_order_5class": GROUP_ORDER,
        "confusion_matrix_12class": cm_12class.tolist(),
        "confusion_matrix_5class": cm_5class.tolist(),
    }

    print("\n========== Test Accuracy ==========")
    print(
        f"12-class accuracy : {accuracy_12class:.4f} "
        f"({accuracy_12class * 100:.2f}%)"
    )
    print(
        f"5-class accuracy  : {accuracy_5class:.4f} "
        f"({accuracy_5class * 100:.2f}%)"
    )
    print(f"Samples           : {len(y_true)}")
    print("===================================\n")

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if OUTPUT_PATH is not None:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
