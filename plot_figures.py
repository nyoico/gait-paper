#!/usr/bin/env python
"""checkpoints/ 의 가중치만으로 confusion matrix와 learning curve를 그립니다.

재학습은 하지 않습니다.

- Confusion matrix: ``checkpoints/`` 안의 가중치를 그대로 불러와 test set을
  한 번만 forward 시켜서 만듭니다. optimizer도, gradient도 쓰지 않습니다.
- Learning curve: 가중치 파일에는 epoch별 기록이 들어 있지 않으므로 학습할 때
  함께 저장해 둔 ``output/history.json``을 읽어서 그립니다. student와 QKD는
  같은 형식의 기록이 ``output/student_fp/history.json``,
  ``output/qkd/history.json``에 따로 있어서 함께 읽습니다.

그림 스타일은 plot_style.py 한 곳에서만 가져옵니다. 이 스크립트는 그림만
만들고 csv/json/npy 같은 수치 파일은 만들지 않습니다.

옵션은 없습니다. 그냥 실행하면 찾을 수 있는 것을 전부 그립니다::

    python plot_figures.py
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import Model as TeacherModel
from qkd_model import QKDStudentModel
from plot_style import (
    CONFUSION_CELL_FONTSIZE,
    CONFUSION_CMAP,
    FIGURE_DPI,
    TICK_FONTSIZE,
    apply_paper_style,
    confusion_text_color,
)
from qkd_common import (
    GROUP_CLASSES,
    VALID_CLASSES,
    confusion_matrix,
    extract_last_logits,
    fine_to_group_indices,
    load_checkpoint,
    make_tgt_inputs,
    prepare_data,
    set_seed,
)


# ---------------------------------------------------------------------------
# 경로와 설정
# ---------------------------------------------------------------------------
PROCESSED_DIR = Path("processed_data")
CHECKPOINT_DIR = Path("checkpoints")
OUTPUT_DIR = Path("figures")

# learning curve의 출처입니다. 학습할 때 저장해 둔 기록을 그대로 그립니다.
HISTORY_DIR = Path("output")
TEACHER_HISTORY = HISTORY_DIR / "history.json"
STUDENT_FP_HISTORY = HISTORY_DIR / "student_fp" / "history.json"
QKD_HISTORY = HISTORY_DIR / "qkd" / "history.json"

# confusion matrix를 만들 split입니다.
SPLIT = "test"
BATCH_SIZE = 64
FIGURE_SUFFIX = ".png"

# 학습 스크립트(train.py, train_student_fp.py, train_qkd.py)와 같은 값이어야
# test set 구성이 그때와 같아집니다.
SEED = 42
VAL_RATIO = 0.03

# train.py의 NUM_HEADS입니다. head 수는 가중치 모양만으로는 알 수 없어서
# config가 없는 체크포인트에만 이 값을 씁니다.
DEFAULT_NUM_HEADS = 8
DEFAULT_DROPOUT = 0.1

# 그림 크기. 기존 train.py / qkd_common.py의 그림과 같게 맞췄습니다.
CONFUSION_FIGSIZE_FINE = (12, 10)
CONFUSION_FIGSIZE_GROUP = (8.5, 7.5)
CURVE_FIGSIZE = (8.5, 6.0)

# 곡선이 아닌 보조선(구간 경계)의 굵기입니다. plot_style의 LINE_WIDTH는
# 데이터 곡선 전용이므로 여기에 쓰지 않습니다.
GUIDE_LINE_WIDTH = 1.2

# 단계 이름을 축 위에 적을 때 제목을 밀어 올리는 정도(pt)입니다.
PHASE_LABEL_TITLE_PAD = 26


# ---------------------------------------------------------------------------
# 체크포인트 목록
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    """checkpoints/ 안의 가중치 하나를 어떻게 복원하는지 적어 둔 표입니다."""

    key: str            # 파일 이름 앞에 붙는 이름
    title: str          # 그림 제목에 쓰는 이름
    filename: str       # checkpoints/ 안의 파일 이름
    kind: str           # "teacher"(model.py) 또는 "student"(qkd_model.py)
    state_key: str      # 체크포인트 안의 state_dict 키
    config_key: Optional[str] = None


MODEL_SPECS: Tuple[ModelSpec, ...] = (
    ModelSpec(
        key="teacher",
        title="Teacher (best)",
        filename="best_model.pt",
        kind="teacher",
        state_key="model_state_dict",
        config_key="config",
    ),
    ModelSpec(
        key="teacher_last",
        title="Teacher (last epoch)",
        filename="last_model.pt",
        kind="teacher",
        state_key="model_state_dict",
        config_key=None,        # last_model.pt에는 config가 없습니다.
    ),
    ModelSpec(
        key="student_fp",
        title="FP32 Student",
        filename="best_student_fp.pt",
        kind="student",
        state_key="model_state_dict",
        config_key="config",
    ),
    ModelSpec(
        key="qkd_ss",
        title="QKD Student (self-studying)",
        filename="best_student_ss.pt",
        kind="student",
        state_key="student_state_dict",
        config_key="student_config",
    ),
    ModelSpec(
        key="qkd_cs",
        title="QKD Student (co-studying)",
        filename="best_qkd_cs.pt",
        kind="student",
        state_key="student_state_dict",
        config_key="student_config",
    ),
    ModelSpec(
        key="qkd_cs_teacher",
        title="QKD Teacher (co-studying)",
        filename="best_qkd_cs.pt",
        kind="teacher",
        state_key="teacher_state_dict",
        config_key=None,
    ),
    ModelSpec(
        key="qkd_tu",
        title="QKD Student (tutoring)",
        filename="best_student_qkd.pt",
        kind="student",
        state_key="student_state_dict",
        config_key="student_config",
    ),
)


# ---------------------------------------------------------------------------
# 모델 복원
# ---------------------------------------------------------------------------
def infer_teacher_config(state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """config가 없는 teacher 체크포인트의 구조를 가중치 모양에서 읽어냅니다.

    num_heads만은 가중치 모양에 남지 않으므로 train.py의 기본값을 씁니다.
    """
    conv = state["sensorL_embed.conv.weight"]
    encoder_layers = {
        int(name.split(".")[1])
        for name in state
        if name.startswith("encoder.")
    }
    return {
        "num_classes": int(state["classifier.weight"].shape[0]),
        "embed_dim": int(conv.shape[0]),
        "sensor_dim": int(conv.shape[1]),
        "num_heads": DEFAULT_NUM_HEADS,
        "num_layers": max(encoder_layers) + 1 if encoder_layers else 1,
        "ff_dim": int(state["encoder.0.ff.0.weight"].shape[0]),
        "dropout": DEFAULT_DROPOUT,
        "max_len": int(state["pos_enc.pe"].shape[1]),
    }


def resolve_config(
    spec: ModelSpec,
    checkpoint: Dict[str, Any],
    state: Dict[str, torch.Tensor],
    shared_teacher_config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """체크포인트에서 모델 생성 인자를 찾습니다.

    1) spec이 가리키는 config 키
    2) 같은 종류의 모델이 흔히 쓰는 config 키
    3) (teacher인 경우) best_model.pt에서 읽어 둔 config
    4) 가중치 모양에서 추론

    best_qkd_cs.pt처럼 teacher와 student가 한 파일에 같이 들어 있는 경우가
    있으므로, 같은 종류의 config 키만 봅니다.
    """
    if spec.kind == "teacher":
        candidates = (spec.config_key, "config")
    else:
        candidates = (spec.config_key, "student_config", "config")

    for key in candidates:
        if key and isinstance(checkpoint.get(key), dict):
            return dict(checkpoint[key])

    if spec.kind == "teacher":
        if shared_teacher_config is not None:
            return dict(shared_teacher_config)
        return infer_teacher_config(state)

    raise KeyError(
        f"{spec.filename}에 student config가 없어 모델을 복원할 수 없습니다."
    )


def build_model(
    spec: ModelSpec,
    checkpoint: Dict[str, Any],
    state: Dict[str, torch.Tensor],
    config: Dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    if spec.kind == "teacher":
        model = TeacherModel(
            num_classes=int(
                config.get("num_classes", state["classifier.weight"].shape[0])
            ),
            embed_dim=int(config["embed_dim"]),
            sensor_dim=int(config["sensor_dim"]),
            num_heads=int(config.get("num_heads", DEFAULT_NUM_HEADS)),
            num_layers=int(config["num_layers"]),
            ff_dim=int(config["ff_dim"]),
            dropout=float(config.get("dropout", DEFAULT_DROPOUT)),
            max_len=int(config.get("max_len", state["pos_enc.pe"].shape[1])),
        )
        model.load_state_dict(state, strict=True)
    else:
        model = QKDStudentModel(**config)
        model.load_state_dict(state, strict=True)
        # LSQ quantizer의 scale과 initialized는 state_dict에 들어 있지만
        # enabled 플래그는 들어 있지 않습니다. 학습 때와 같은 상태로 되돌립니다.
        if bool(checkpoint.get("quantization_enabled", False)):
            model.enable_quantization()
        else:
            model.disable_quantization()

    return model.to(device).eval()


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """forward만 돌려 정답과 예측을 모읍니다. 학습은 하지 않습니다."""
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
        all_true.extend(labels.cpu().tolist())
        all_pred.extend(logits.argmax(dim=1).cpu().tolist())

    return (
        np.asarray(all_true, dtype=np.int64),
        np.asarray(all_pred, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# 데이터 준비 (체크포인트마다 정규화 통계가 다를 수 있어 캐시해 둡니다)
# ---------------------------------------------------------------------------
_DATA_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}


def _array_key(mapping: Optional[Dict[str, Any]], dtype) -> str:
    if not mapping:
        return "auto"
    digest = hashlib.sha1()
    for name in sorted(mapping):
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(mapping[name], dtype=dtype).tobytes())
    return digest.hexdigest()


def get_data(
    normalization: Optional[Dict[str, Any]],
    split_indices: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    key = (
        _array_key(normalization, np.float32),
        _array_key(split_indices, np.int64),
    )
    if key not in _DATA_CACHE:
        print(f"[data] {PROCESSED_DIR} 를 읽는 중입니다 ...", flush=True)
        _DATA_CACHE[key] = prepare_data(
            processed_dir=PROCESSED_DIR,
            batch_size=BATCH_SIZE,
            num_workers=0,
            val_ratio=VAL_RATIO,
            seed=SEED,
            split_indices=split_indices,
            normalization=normalization,
        )
    return _DATA_CACHE[key]


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
def plot_confusion(
    matrix: np.ndarray,
    labels: Sequence[str],
    title: str,
    path: Path,
    normalize: bool = False,
) -> None:
    """블루 계열 confusion matrix 한 장을 저장합니다.

    plot_style의 약속대로 컬러바는 쓰지 않고 셀 안에 값을 적습니다.
    """
    values = matrix.astype(np.float64)
    if normalize:
        row_sums = values.sum(axis=1, keepdims=True)
        values = np.divide(
            values * 100.0,
            row_sums,
            out=np.zeros_like(values),
            where=row_sums > 0,
        )
        vmax = 100.0
    else:
        vmax = float(values.max()) if values.size else 0.0

    figsize = (
        CONFUSION_FIGSIZE_FINE if len(labels) > 5 else CONFUSION_FIGSIZE_GROUP
    )
    plt.figure(figsize=figsize)
    plt.imshow(values, cmap=CONFUSION_CMAP, vmin=0.0, vmax=max(vmax, 1e-9))
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(np.arange(len(labels)), labels)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if normalize:
                if value <= 0.0:
                    continue    # 비율 그림에서 0.0을 다 적으면 읽기 어렵습니다.
                text = f"{value:.1f}"
            else:
                text = str(int(matrix[i, j]))
            plt.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=CONFUSION_CELL_FONTSIZE,
                color=confusion_text_color(value, vmax),
            )

    plt.tight_layout()
    plt.savefig(path, dpi=FIGURE_DPI)
    plt.close()


def draw_confusion_figures(
    spec: ModelSpec,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> List[Path]:
    """12-class와 5-class를, 개수판과 백분율판으로 각각 저장합니다."""
    group_true = fine_to_group_indices(y_true)
    group_pred = fine_to_group_indices(y_pred)

    panels = (
        (
            "12class",
            "12-class",
            confusion_matrix(y_true, y_pred, len(VALID_CLASSES)),
            VALID_CLASSES,
        ),
        (
            "5class",
            "5-class",
            confusion_matrix(group_true, group_pred, len(GROUP_CLASSES)),
            GROUP_CLASSES,
        ),
    )

    written: List[Path] = []
    for stem, description, matrix, labels in panels:
        counts_path = OUTPUT_DIR / f"{spec.key}_confusion_{stem}{FIGURE_SUFFIX}"
        plot_confusion(
            matrix,
            labels,
            f"{spec.title} - {description} ({SPLIT})",
            counts_path,
            normalize=False,
        )
        written.append(counts_path)

        percent_path = (
            OUTPUT_DIR / f"{spec.key}_confusion_{stem}_percent{FIGURE_SUFFIX}"
        )
        plot_confusion(
            matrix,
            labels,
            f"{spec.title} - {description} ({SPLIT}, %)",
            percent_path,
            normalize=True,
        )
        written.append(percent_path)

    fine_accuracy = float((y_true == y_pred).mean()) if y_true.size else 0.0
    group_accuracy = (
        float((group_true == group_pred).mean()) if group_true.size else 0.0
    )
    print(
        f"  {SPLIT} accuracy: 12-class {fine_accuracy:.4f} / "
        f"5-class {group_accuracy:.4f}"
    )
    return written


def draw_all_confusions(specs: Sequence[ModelSpec]) -> List[Path]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    written: List[Path] = []

    # config가 없는 teacher 체크포인트(last_model.pt, co-studying teacher)를
    # 위해 best_model.pt의 config를 미리 읽어 둡니다.
    shared_teacher_config: Optional[Dict[str, Any]] = None
    teacher_best = CHECKPOINT_DIR / "best_model.pt"
    if teacher_best.exists():
        shared_teacher_config = load_checkpoint(
            teacher_best, map_location="cpu"
        ).get("config")

    for spec in specs:
        path = CHECKPOINT_DIR / spec.filename
        print(f"[{spec.key}] {path}")
        checkpoint = load_checkpoint(path, map_location="cpu")

        state = checkpoint.get(spec.state_key)
        if not isinstance(state, dict):
            print(f"  [skip] '{spec.state_key}' 가 체크포인트에 없습니다.")
            continue

        try:
            config = resolve_config(spec, checkpoint, state, shared_teacher_config)
            model = build_model(spec, checkpoint, state, config, device)
        except (KeyError, RuntimeError, TypeError) as error:
            print(f"  [skip] 모델을 복원하지 못했습니다: {error}")
            continue

        try:
            data = get_data(
                checkpoint.get("normalization"),
                checkpoint.get("split_indices"),
            )
        except FileNotFoundError as error:
            print(f"  [stop] 데이터를 찾지 못했습니다: {error}")
            del model
            break

        y_true, y_pred = predict(model, data[f"{SPLIT}_loader"], device)
        written += draw_confusion_figures(spec, y_true, y_pred)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return written


# ---------------------------------------------------------------------------
# Learning curve
# ---------------------------------------------------------------------------
@dataclass
class Curve:
    label: str
    values: List[float]
    linestyle: str = "-"


@dataclass
class PhaseSpan:
    """QKD처럼 여러 단계로 나뉜 학습에서 한 단계가 차지하는 epoch 구간입니다."""

    start: float
    end: float
    label: str


@dataclass
class CurveSet:
    key: str
    title: str
    epochs: List[int]
    loss: List[Curve] = field(default_factory=list)
    acc12: List[Curve] = field(default_factory=list)
    acc5: List[Curve] = field(default_factory=list)
    phases: List[PhaseSpan] = field(default_factory=list)


def _clean(values: Sequence[Any]) -> List[float]:
    """None을 nan으로 바꿔서 기록이 없는 구간은 그대로 비워 둡니다."""
    return [float("nan") if value is None else float(value) for value in values]


def _has_data(values: Sequence[float]) -> bool:
    return any(not math.isnan(value) for value in values)


def plot_curve(
    epochs: Sequence[int],
    curves: Sequence[Curve],
    ylabel: str,
    title: str,
    path: Path,
    phases: Sequence[PhaseSpan] = (),
) -> None:
    figure, axes = plt.subplots(figsize=CURVE_FIGSIZE)

    for curve in curves:
        axes.plot(
            epochs,
            curve.values,
            label=curve.label,
            linestyle=curve.linestyle,
        )

    axes.set_xlabel("Epoch")
    axes.set_ylabel(ylabel)
    # 단계 이름을 축 바로 위에 적으므로 제목을 그만큼 밀어 올립니다.
    axes.set_title(title, pad=PHASE_LABEL_TITLE_PAD if phases else None)
    axes.legend()

    # 단계 경계는 얇은 회색 점선으로만 그려 데이터 곡선과 구분하고, 이름은
    # 축 바깥 위쪽에 적어 곡선이나 범례와 겹치지 않게 합니다.
    for index, phase in enumerate(phases):
        if index > 0:
            axes.axvline(
                phase.start - 0.5,
                color="0.55",
                linestyle="--",
                linewidth=GUIDE_LINE_WIDTH,
                zorder=0,
            )
        axes.text(
            (phase.start + phase.end) / 2.0,
            1.015,
            phase.label,
            transform=axes.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=TICK_FONTSIZE,
            color="0.35",
        )

    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)


def draw_curve_figures(curve_set: CurveSet) -> List[Path]:
    written: List[Path] = []
    panels = (
        (curve_set.loss, "Loss", "loss", "learning_curve_loss"),
        (
            curve_set.acc12,
            "Accuracy",
            "accuracy (12 classes)",
            "learning_curve_acc_12class",
        ),
        (
            curve_set.acc5,
            "Accuracy",
            "accuracy (5 classes)",
            "learning_curve_acc_5class",
        ),
    )

    for curves, ylabel, description, stem in panels:
        usable = [curve for curve in curves if _has_data(curve.values)]
        if not usable:
            continue
        path = OUTPUT_DIR / f"{curve_set.key}_{stem}{FIGURE_SUFFIX}"
        plot_curve(
            curve_set.epochs,
            usable,
            ylabel,
            f"{curve_set.title} - {description}",
            path,
            phases=curve_set.phases,
        )
        written.append(path)

    return written


def read_teacher_history(path: Path) -> CurveSet:
    """output/history.json입니다. train.py가 epoch마다 남긴 기록입니다.

    예전 실행본은 train_acc/val_acc, 최근 실행본은 train_fine_acc/
    train_coarse_acc를 씁니다. 둘 다 받습니다.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))

    def column(*names: str) -> List[float]:
        for name in names:
            if name in raw:
                return _clean(raw[name])
        return []

    train_loss = column("train_loss")
    return CurveSet(
        key="teacher",
        title="Teacher",
        epochs=list(range(1, len(train_loss) + 1)),
        loss=[
            Curve("train loss", train_loss),
            Curve("val loss", column("val_loss")),
        ],
        acc12=[
            Curve("train acc", column("train_fine_acc", "train_acc")),
            Curve("val acc", column("val_fine_acc", "val_acc")),
        ],
        acc5=[
            Curve("train acc", column("train_coarse_acc")),
            Curve("val acc", column("val_coarse_acc")),
        ],
    )


def read_student_fp_history(path: Path) -> CurveSet:
    """train_student_fp.py가 남긴 epoch별 행 목록입니다."""
    rows = json.loads(path.read_text(encoding="utf-8"))

    def column(name: str) -> List[float]:
        return _clean([row.get(name) for row in rows])

    return CurveSet(
        key="student_fp",
        title="FP32 Student",
        epochs=[
            int(row.get("epoch", index))
            for index, row in enumerate(rows, start=1)
        ],
        loss=[
            Curve("train loss", column("train_loss")),
            Curve("val loss", column("val_loss")),
        ],
        acc12=[
            Curve("train acc", column("train_accuracy_12class")),
            Curve("val acc", column("val_accuracy_12class")),
        ],
        acc5=[
            Curve("train acc", column("train_accuracy_5class")),
            Curve("val acc", column("val_accuracy_5class")),
        ],
    )


QKD_PHASES: Tuple[Tuple[str, str], ...] = (
    ("ss", "Self-studying"),
    ("cs", "Co-studying"),
    ("tu", "Tutoring"),
)

TEACHER_SERIES = (
    "teacher_train_loss", "teacher_val_loss",
    "teacher_train_acc12", "teacher_val_acc12",
    "teacher_train_acc5", "teacher_val_acc5",
)


def read_qkd_history(path: Path) -> CurveSet:
    """train_qkd.py가 남긴 3단계 기록을 한 축 위에 이어 붙입니다.

    단계마다 손실 정의가 다릅니다(SS는 CE, CS/TU는 CE + T^2*KL). 그래서 손실
    그림에는 단계 경계선을 함께 그려 구간별로 읽도록 합니다.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))

    epochs: List[int] = []
    phases: List[PhaseSpan] = []
    series: Dict[str, List[Any]] = {
        name: []
        for name in (
            "train_loss", "val_loss",
            "train_acc12", "val_acc12",
            "train_acc5", "val_acc5",
        ) + TEACHER_SERIES
    }

    offset = 0
    for name, label in QKD_PHASES:
        rows = raw.get(name) or []
        if not rows:
            continue
        phases.append(
            PhaseSpan(start=offset + 1, end=offset + len(rows), label=label)
        )
        for index, row in enumerate(rows, start=1):
            epochs.append(offset + index)
            train = row.get("train", {})
            if name == "cs":
                # co-studying은 student와 teacher 기록이 나뉘어 있습니다.
                series["train_loss"].append(train.get("student_loss"))
                series["val_loss"].append(row.get("student_val_loss"))
                series["train_acc12"].append(
                    row.get("train_student_accuracy_12class")
                )
                series["val_acc12"].append(
                    row.get("student_val_accuracy_12class")
                )
                series["train_acc5"].append(
                    row.get("train_student_accuracy_5class")
                )
                series["val_acc5"].append(row.get("student_val_accuracy_5class"))
                series["teacher_train_loss"].append(train.get("teacher_loss"))
                series["teacher_val_loss"].append(row.get("teacher_val_loss"))
                series["teacher_train_acc12"].append(
                    row.get("train_teacher_accuracy_12class")
                )
                series["teacher_val_acc12"].append(
                    row.get("teacher_val_accuracy_12class")
                )
                series["teacher_train_acc5"].append(
                    row.get("train_teacher_accuracy_5class")
                )
                series["teacher_val_acc5"].append(
                    row.get("teacher_val_accuracy_5class")
                )
            else:
                series["train_loss"].append(train.get("loss"))
                series["val_loss"].append(row.get("val_loss"))
                series["train_acc12"].append(
                    row.get("train_accuracy_12class", train.get("fine_accuracy"))
                )
                series["val_acc12"].append(
                    row.get("val_accuracy_12class", row.get("val_accuracy"))
                )
                series["train_acc5"].append(
                    row.get("train_accuracy_5class", train.get("group_accuracy"))
                )
                series["val_acc5"].append(row.get("val_accuracy_5class"))
                # teacher는 co-studying에서만 움직이므로 나머지 구간은 비웁니다.
                for teacher_name in TEACHER_SERIES:
                    series[teacher_name].append(None)
        offset += len(rows)

    return CurveSet(
        key="qkd",
        title="QKD Student",
        epochs=epochs,
        loss=[
            Curve("student train loss", _clean(series["train_loss"])),
            Curve("student val loss", _clean(series["val_loss"])),
            Curve("teacher train loss", _clean(series["teacher_train_loss"]), ":"),
            Curve("teacher val loss", _clean(series["teacher_val_loss"]), ":"),
        ],
        acc12=[
            Curve("student train acc", _clean(series["train_acc12"])),
            Curve("student val acc", _clean(series["val_acc12"])),
            Curve("teacher train acc", _clean(series["teacher_train_acc12"]), ":"),
            Curve("teacher val acc", _clean(series["teacher_val_acc12"]), ":"),
        ],
        acc5=[
            Curve("student train acc", _clean(series["train_acc5"])),
            Curve("student val acc", _clean(series["val_acc5"])),
            Curve("teacher train acc", _clean(series["teacher_train_acc5"]), ":"),
            Curve("teacher val acc", _clean(series["teacher_val_acc5"]), ":"),
        ],
        phases=phases,
    )


def draw_all_curves() -> List[Path]:
    """output/ 아래의 history.json을 모두 읽어 learning curve를 그립니다."""
    sources = (
        (TEACHER_HISTORY, read_teacher_history),
        (STUDENT_FP_HISTORY, read_student_fp_history),
        (QKD_HISTORY, read_qkd_history),
    )

    written: List[Path] = []
    for path, reader in sources:
        if not path.exists():
            print(f"[skip] history 파일이 없습니다: {path}")
            continue
        try:
            curve_set = reader(path)
        except (KeyError, TypeError, ValueError) as error:
            print(f"[skip] {path} 를 읽지 못했습니다: {error}")
            continue
        print(
            f"[{curve_set.key}] {path} "
            f"({len(curve_set.epochs)} epochs)"
        )
        written += draw_curve_figures(curve_set)
    return written


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def main() -> int:
    apply_paper_style()
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    specs = [
        spec
        for spec in MODEL_SPECS
        if (CHECKPOINT_DIR / spec.filename).exists()
    ]
    if not specs:
        print(f"{CHECKPOINT_DIR} 에서 쓸 수 있는 체크포인트를 찾지 못했습니다.")

    written = draw_all_confusions(specs) + draw_all_curves()

    print(f"\n{len(written)}개의 그림을 {OUTPUT_DIR} 에 저장했습니다.")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
