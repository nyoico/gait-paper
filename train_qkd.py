import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from model_teacher_original import Model as TeacherModel
from qkd_common import (
    FINE_TO_GROUP,
    GROUP_LABEL_TO_IDX,
    LABEL_TO_IDX,
    load_checkpoint,
    VALID_CLASSES,
    count_parameters,
    evaluate_model,
    extract_last_logits,
    make_tgt_inputs,
    measure_latency,
    prepare_data,
    save_evaluation_outputs,
    set_seed,
)
from qkd_model import QKDStudentModel


SEED = 42
PROCESSED_DIR = Path("processed_data")
CHECKPOINT_DIR = Path("checkpoints")
OUTPUT_DIR = Path("output/qkd")

# 12개 세부 클래스 인덱스를 5개 상위 그룹 인덱스로 옮기는 표.
FINE_TO_GROUP_INDEX = torch.tensor(
    [GROUP_LABEL_TO_IDX[FINE_TO_GROUP[name]] for name in VALID_CLASSES],
    dtype=torch.long,
)


def count_group_correct(logits: torch.Tensor, labels: torch.Tensor) -> int:
    """예측과 정답을 5개 상위 그룹으로 합친 뒤 맞은 개수를 센다."""
    table = FINE_TO_GROUP_INDEX.to(labels.device)
    return int((table[logits.argmax(dim=1)] == table[labels]).sum().item())

TEACHER_CHECKPOINT = CHECKPOINT_DIR / "best_model.pt"
FP_STUDENT_CHECKPOINT = CHECKPOINT_DIR / "best_student_fp.pt"

BATCH_SIZE = 32
NUM_WORKERS = 0
VAL_RATIO = 0.03
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
TEMPERATURE = 2.0

SS_EPOCHS = 30
CS_EPOCHS = 30
TU_EPOCHS = 40

SS_LR = 1e-4
CS_STUDENT_LR = 1e-4
CS_TEACHER_LR = 1e-5
TU_LR = 5e-5

WEIGHT_INTERVAL_LR_RATIO = 0.01  # QKD paper: 100x smaller than weight LR.
ACTIVATION_INTERVAL_LR_RATIO = 1.0


def student_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    ce = F.cross_entropy(student_logits, labels, label_smoothing=0.0)
    kl = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits.detach() / temperature, dim=1),
        reduction="batchmean",
    )
    total = ce + (temperature**2) * kl
    return total, {"ce": ce.item(), "kl": kl.item(), "total": total.item()}


def teacher_kd_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    ce = F.cross_entropy(teacher_logits, labels, label_smoothing=0.0)
    kl = F.kl_div(
        F.log_softmax(teacher_logits / temperature, dim=1),
        F.softmax(student_logits.detach() / temperature, dim=1),
        reduction="batchmean",
    )
    total = ce + (temperature**2) * kl
    return total, {"ce": ce.item(), "kl": kl.item(), "total": total.item()}


def build_student_optimizer(
    student: QKDStudentModel,
    base_lr: float,
) -> torch.optim.Optimizer:
    normal_parameters = []
    weight_interval_parameters = []
    activation_interval_parameters = []

    for name, parameter in student.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith("weight_quant.scale"):
            weight_interval_parameters.append(parameter)
        elif name.endswith("activation_quant.scale") or name.endswith("output_quant.scale"):
            activation_interval_parameters.append(parameter)
        else:
            normal_parameters.append(parameter)

    groups = [
        {
            "params": normal_parameters,
            "lr": base_lr,
            "weight_decay": WEIGHT_DECAY,
            "group_name": "model",
        },
        {
            "params": weight_interval_parameters,
            "lr": base_lr * WEIGHT_INTERVAL_LR_RATIO,
            "weight_decay": 0.0,
            "group_name": "weight_interval",
        },
        {
            "params": activation_interval_parameters,
            "lr": base_lr * ACTIVATION_INTERVAL_LR_RATIO,
            "weight_decay": 0.0,
            "group_name": "activation_interval",
        },
    ]
    return torch.optim.AdamW(groups)


def make_scheduler(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
        min_lr=1e-7,
    )


def self_studying_epoch(student, loader, optimizer, device):
    student.train()
    total_loss = 0.0
    total_correct = 0
    total_group_correct = 0
    total_count = 0

    for left, right, labels in loader:
        left = left.to(device, non_blocking=True)
        right = right.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        tgt_seq, tgt_mask = make_tgt_inputs(labels.size(0), device)

        logits = extract_last_logits(
            student(left, right, tgt_seq, tgt_mask=tgt_mask)
        )
        loss = F.cross_entropy(logits, labels, label_smoothing=0.0)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_group_correct += count_group_correct(logits, labels)
        total_count += labels.size(0)

    fine_accuracy = total_correct / total_count
    group_accuracy = total_group_correct / total_count
    return {
        "loss": total_loss / total_count,
        # `accuracy`는 기존 키 유지용이며 12-class 정확도다.
        "accuracy": fine_accuracy,
        "fine_accuracy": fine_accuracy,
        "group_accuracy": group_accuracy,
        "accuracy_12class": fine_accuracy,
        "accuracy_5class": group_accuracy,
    }


def co_studying_epoch(
    teacher,
    student,
    loader,
    teacher_optimizer,
    student_optimizer,
    device,
):
    teacher.train()
    student.train()

    sums = {
        "student_loss": 0.0,
        "teacher_loss": 0.0,
        "student_ce": 0.0,
        "teacher_ce": 0.0,
        "student_kl": 0.0,
        "teacher_kl": 0.0,
        "student_correct": 0,
        "teacher_correct": 0,
        "student_group_correct": 0,
        "teacher_group_correct": 0,
        "count": 0,
    }

    for left, right, labels in loader:
        left = left.to(device, non_blocking=True)
        right = right.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        tgt_seq, tgt_mask = make_tgt_inputs(labels.size(0), device)

        teacher_logits = extract_last_logits(
            teacher(left, right, tgt_seq, tgt_mask=tgt_mask)
        )
        student_logits = extract_last_logits(
            student(left, right, tgt_seq, tgt_mask=tgt_mask)
        )

        s_loss, s_parts = student_kd_loss(
            student_logits,
            teacher_logits,
            labels,
            TEMPERATURE,
        )
        t_loss, t_parts = teacher_kd_loss(
            teacher_logits,
            student_logits,
            labels,
            TEMPERATURE,
        )

        teacher_optimizer.zero_grad(set_to_none=True)
        student_optimizer.zero_grad(set_to_none=True)
        (s_loss + t_loss).backward()
        torch.nn.utils.clip_grad_norm_(teacher.parameters(), GRAD_CLIP)
        torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
        teacher_optimizer.step()
        student_optimizer.step()

        batch_size = labels.size(0)
        sums["student_loss"] += s_parts["total"] * batch_size
        sums["teacher_loss"] += t_parts["total"] * batch_size
        sums["student_ce"] += s_parts["ce"] * batch_size
        sums["teacher_ce"] += t_parts["ce"] * batch_size
        sums["student_kl"] += s_parts["kl"] * batch_size
        sums["teacher_kl"] += t_parts["kl"] * batch_size
        sums["student_correct"] += (
            student_logits.argmax(dim=1) == labels
        ).sum().item()
        sums["teacher_correct"] += (
            teacher_logits.argmax(dim=1) == labels
        ).sum().item()
        sums["student_group_correct"] += count_group_correct(
            student_logits, labels
        )
        sums["teacher_group_correct"] += count_group_correct(
            teacher_logits, labels
        )
        sums["count"] += batch_size

    count = sums.pop("count")
    return {
        "student_loss": sums["student_loss"] / count,
        "teacher_loss": sums["teacher_loss"] / count,
        "student_ce": sums["student_ce"] / count,
        "teacher_ce": sums["teacher_ce"] / count,
        "student_kl": sums["student_kl"] / count,
        "teacher_kl": sums["teacher_kl"] / count,
        "student_accuracy": sums["student_correct"] / count,
        "teacher_accuracy": sums["teacher_correct"] / count,
        "student_group_accuracy": sums["student_group_correct"] / count,
        "teacher_group_accuracy": sums["teacher_group_correct"] / count,
    }


def tutoring_epoch(teacher, student, loader, student_optimizer, device):
    teacher.eval()
    student.train()
    total_loss = 0.0
    total_ce = 0.0
    total_kl = 0.0
    total_correct = 0
    total_group_correct = 0
    total_count = 0

    for left, right, labels in loader:
        left = left.to(device, non_blocking=True)
        right = right.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        tgt_seq, tgt_mask = make_tgt_inputs(labels.size(0), device)

        with torch.no_grad():
            teacher_logits = extract_last_logits(
                teacher(left, right, tgt_seq, tgt_mask=tgt_mask)
            )

        student_logits = extract_last_logits(
            student(left, right, tgt_seq, tgt_mask=tgt_mask)
        )
        loss, parts = student_kd_loss(
            student_logits,
            teacher_logits,
            labels,
            TEMPERATURE,
        )

        student_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
        student_optimizer.step()

        batch_size = labels.size(0)
        total_loss += parts["total"] * batch_size
        total_ce += parts["ce"] * batch_size
        total_kl += parts["kl"] * batch_size
        total_correct += (
            student_logits.argmax(dim=1) == labels
        ).sum().item()
        total_group_correct += count_group_correct(student_logits, labels)
        total_count += batch_size

    fine_accuracy = total_correct / total_count
    group_accuracy = total_group_correct / total_count
    return {
        "loss": total_loss / total_count,
        "ce": total_ce / total_count,
        "kl": total_kl / total_count,
        # `accuracy`는 기존 키 유지용이며 12-class 정확도다.
        "accuracy": fine_accuracy,
        "fine_accuracy": fine_accuracy,
        "group_accuracy": group_accuracy,
        "accuracy_12class": fine_accuracy,
        "accuracy_5class": group_accuracy,
    }


def load_teacher(device):
    if not TEACHER_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {TEACHER_CHECKPOINT}"
        )
    checkpoint = load_checkpoint(TEACHER_CHECKPOINT, map_location=device)
    config = checkpoint.get("config", {})
    teacher = TeacherModel(
        num_classes=len(VALID_CLASSES),
        embed_dim=int(config.get("embed_dim", 256)),
        sensor_dim=int(config.get("sensor_dim", 5)),
        num_heads=int(config.get("num_heads", 8)),
        num_layers=int(config.get("num_layers", 4)),
        ff_dim=int(config.get("ff_dim", 1024)),
        dropout=float(config.get("dropout", 0.1)),
        max_len=int(config.get("max_len", 101)),
    ).to(device)
    teacher.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return teacher, checkpoint


def load_fp_student(device):
    if not FP_STUDENT_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"FP student checkpoint not found: {FP_STUDENT_CHECKPOINT}. "
            "Run train_student_fp.py first."
        )
    checkpoint = load_checkpoint(FP_STUDENT_CHECKPOINT, map_location=device)
    config = checkpoint["config"]
    student = QKDStudentModel(**config).to(device)
    student.load_state_dict(checkpoint["model_state_dict"], strict=True)
    student.reset_quantizers()
    student.enable_quantization()
    return student, checkpoint


def save_phase_checkpoint(
    path: Path,
    stage: str,
    epoch: int,
    student: QKDStudentModel,
    student_val_acc: float,
    student_config: Dict,
    data: Dict,
    teacher=None,
    teacher_val_acc=None,
):
    payload = {
        "stage": stage,
        "epoch": epoch,
        "student_state_dict": student.state_dict(),
        "student_config": student_config,
        "student_val_acc": student_val_acc,
        "label_to_idx": LABEL_TO_IDX,
        "normalization": data["normalization"],
        "split_indices": data["split_indices"],
        "quantization_enabled": True,
        "quantizer_summary": student.quantizer_summary(),
    }
    if teacher is not None:
        payload["teacher_state_dict"] = teacher.state_dict()
        payload["teacher_val_acc"] = teacher_val_acc
    torch.save(payload, path)


def main():
    set_seed(SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    teacher, teacher_checkpoint = load_teacher(device)
    student, fp_checkpoint = load_fp_student(device)

    teacher_mapping = teacher_checkpoint.get("label_to_idx")
    if teacher_mapping is not None and teacher_mapping != LABEL_TO_IDX:
        raise ValueError(
            "Teacher class order differs from QKD class order. "
            f"teacher={teacher_mapping}, expected={LABEL_TO_IDX}"
        )

    data = prepare_data(
        processed_dir=PROCESSED_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        val_ratio=VAL_RATIO,
        seed=SEED,
        split_indices=fp_checkpoint["split_indices"],
        normalization=fp_checkpoint["normalization"],
    )

    print("Teacher parameters:", count_parameters(teacher))
    student_params = count_parameters(student)
    print("Student parameters:", student_params)
    if student_params["total"] > 5_000_000:
        raise RuntimeError(f"Student exceeds 5M parameters: {student_params['total']:,}")

    history = {"ss": [], "cs": [], "tu": []}
    # SS/CS/TU 단계별 정확도를 12-class와 5-class로 함께 모은다.
    stage_accuracy = {}

    # -------------------- Phase 1: Self-studying --------------------
    ss_optimizer = build_student_optimizer(student, SS_LR)
    ss_scheduler = make_scheduler(ss_optimizer)
    ss_path = CHECKPOINT_DIR / "best_student_ss.pt"
    best_ss_acc = -1.0
    best_ss_group_acc = -1.0

    for epoch in tqdm(range(1, SS_EPOCHS + 1), desc="QKD Phase 1 / SS"):
        train_result = self_studying_epoch(
            student,
            data["train_loader"],
            ss_optimizer,
            device,
        )
        val_result = evaluate_model(student, data["val_loader"], device)
        ss_scheduler.step(val_result["accuracy"])

        row = {
            "epoch": epoch,
            "train": train_result,
            "val_loss": val_result["loss"],
            "val_accuracy": val_result["accuracy"],
            "train_accuracy_12class": train_result["fine_accuracy"],
            "train_accuracy_5class": train_result["group_accuracy"],
            "val_accuracy_12class": val_result["fine_accuracy"],
            "val_accuracy_5class": val_result["group_accuracy"],
            "lrs": [group["lr"] for group in ss_optimizer.param_groups],
        }
        history["ss"].append(row)
        tqdm.write(
            f"[SS {epoch:03d}] "
            f"train acc(12/5)={train_result['fine_accuracy']:.4f}/"
            f"{train_result['group_accuracy']:.4f} | "
            f"val acc(12/5)={val_result['fine_accuracy']:.4f}/"
            f"{val_result['group_accuracy']:.4f}"
        )

        if val_result["accuracy"] > best_ss_acc:
            best_ss_acc = val_result["accuracy"]
            best_ss_group_acc = val_result["group_accuracy"]
            save_phase_checkpoint(
                ss_path,
                "self_studying",
                epoch,
                student,
                best_ss_acc,
                student.config,
                data,
            )

    ss_checkpoint = load_checkpoint(ss_path, map_location=device)
    student.load_state_dict(ss_checkpoint["student_state_dict"])
    student.enable_quantization()

    ss_test = evaluate_model(student, data["test_loader"], device)
    stage_accuracy["ss"] = {
        "best_val_accuracy_12class": best_ss_acc,
        "best_val_accuracy_5class": best_ss_group_acc,
        "test_accuracy_12class": ss_test["fine_accuracy"],
        "test_accuracy_5class": ss_test["group_accuracy"],
    }
    print(
        f"[SS done] best val acc(12/5)={best_ss_acc:.4f}/"
        f"{best_ss_group_acc:.4f} | "
        f"test acc(12/5)={ss_test['fine_accuracy']:.4f}/"
        f"{ss_test['group_accuracy']:.4f}"
    )

    # -------------------- Phase 2: Co-studying --------------------
    cs_student_optimizer = build_student_optimizer(student, CS_STUDENT_LR)
    cs_teacher_optimizer = torch.optim.AdamW(
        teacher.parameters(),
        lr=CS_TEACHER_LR,
        weight_decay=WEIGHT_DECAY,
    )
    cs_student_scheduler = make_scheduler(cs_student_optimizer)
    cs_teacher_scheduler = make_scheduler(cs_teacher_optimizer)
    cs_path = CHECKPOINT_DIR / "best_qkd_cs.pt"
    best_cs_acc = -1.0
    best_cs_group_acc = -1.0
    best_cs_teacher_acc = -1.0
    best_cs_teacher_group_acc = -1.0

    for epoch in tqdm(range(1, CS_EPOCHS + 1), desc="QKD Phase 2 / CS"):
        train_result = co_studying_epoch(
            teacher,
            student,
            data["train_loader"],
            cs_teacher_optimizer,
            cs_student_optimizer,
            device,
        )
        student_val = evaluate_model(student, data["val_loader"], device)
        teacher_val = evaluate_model(teacher, data["val_loader"], device)
        cs_student_scheduler.step(student_val["accuracy"])
        cs_teacher_scheduler.step(teacher_val["accuracy"])

        row = {
            "epoch": epoch,
            "train": train_result,
            "student_val_loss": student_val["loss"],
            "student_val_accuracy": student_val["accuracy"],
            "teacher_val_loss": teacher_val["loss"],
            "teacher_val_accuracy": teacher_val["accuracy"],
            "train_student_accuracy_12class": train_result["student_accuracy"],
            "train_student_accuracy_5class": train_result[
                "student_group_accuracy"
            ],
            "train_teacher_accuracy_12class": train_result["teacher_accuracy"],
            "train_teacher_accuracy_5class": train_result[
                "teacher_group_accuracy"
            ],
            "student_val_accuracy_12class": student_val["fine_accuracy"],
            "student_val_accuracy_5class": student_val["group_accuracy"],
            "teacher_val_accuracy_12class": teacher_val["fine_accuracy"],
            "teacher_val_accuracy_5class": teacher_val["group_accuracy"],
        }
        history["cs"].append(row)
        tqdm.write(
            f"[CS {epoch:03d}] "
            f"student val acc(12/5)={student_val['fine_accuracy']:.4f}/"
            f"{student_val['group_accuracy']:.4f} | "
            f"teacher val acc(12/5)={teacher_val['fine_accuracy']:.4f}/"
            f"{teacher_val['group_accuracy']:.4f} | "
            f"S_KL={train_result['student_kl']:.4f}"
        )

        if student_val["accuracy"] > best_cs_acc:
            best_cs_acc = student_val["accuracy"]
            best_cs_group_acc = student_val["group_accuracy"]
            best_cs_teacher_acc = teacher_val["fine_accuracy"]
            best_cs_teacher_group_acc = teacher_val["group_accuracy"]
            save_phase_checkpoint(
                cs_path,
                "co_studying",
                epoch,
                student,
                best_cs_acc,
                student.config,
                data,
                teacher=teacher,
                teacher_val_acc=teacher_val["accuracy"],
            )

    cs_checkpoint = load_checkpoint(cs_path, map_location=device)
    student.load_state_dict(cs_checkpoint["student_state_dict"])
    student.enable_quantization()
    teacher.load_state_dict(cs_checkpoint["teacher_state_dict"])

    cs_test = evaluate_model(student, data["test_loader"], device)
    cs_teacher_test = evaluate_model(teacher, data["test_loader"], device)
    stage_accuracy["cs"] = {
        "best_val_accuracy_12class": best_cs_acc,
        "best_val_accuracy_5class": best_cs_group_acc,
        "test_accuracy_12class": cs_test["fine_accuracy"],
        "test_accuracy_5class": cs_test["group_accuracy"],
        "teacher_val_accuracy_12class_at_best_student": best_cs_teacher_acc,
        "teacher_val_accuracy_5class_at_best_student": (
            best_cs_teacher_group_acc
        ),
        "teacher_test_accuracy_12class": cs_teacher_test["fine_accuracy"],
        "teacher_test_accuracy_5class": cs_teacher_test["group_accuracy"],
    }
    print(
        f"[CS done] student best val acc(12/5)={best_cs_acc:.4f}/"
        f"{best_cs_group_acc:.4f} | "
        f"student test acc(12/5)={cs_test['fine_accuracy']:.4f}/"
        f"{cs_test['group_accuracy']:.4f} | "
        f"teacher test acc(12/5)={cs_teacher_test['fine_accuracy']:.4f}/"
        f"{cs_teacher_test['group_accuracy']:.4f}"
    )

    # -------------------- Phase 3: Tutoring --------------------
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    tu_optimizer = build_student_optimizer(student, TU_LR)
    tu_scheduler = make_scheduler(tu_optimizer)
    final_path = CHECKPOINT_DIR / "best_student_qkd.pt"
    best_tu_acc = -1.0
    best_tu_group_acc = -1.0

    for epoch in tqdm(range(1, TU_EPOCHS + 1), desc="QKD Phase 3 / TU"):
        train_result = tutoring_epoch(
            teacher,
            student,
            data["train_loader"],
            tu_optimizer,
            device,
        )
        val_result = evaluate_model(student, data["val_loader"], device)
        tu_scheduler.step(val_result["accuracy"])

        row = {
            "epoch": epoch,
            "train": train_result,
            "val_loss": val_result["loss"],
            "val_accuracy": val_result["accuracy"],
            "train_accuracy_12class": train_result["fine_accuracy"],
            "train_accuracy_5class": train_result["group_accuracy"],
            "val_accuracy_12class": val_result["fine_accuracy"],
            "val_accuracy_5class": val_result["group_accuracy"],
        }
        history["tu"].append(row)
        tqdm.write(
            f"[TU {epoch:03d}] "
            f"train acc(12/5)={train_result['fine_accuracy']:.4f}/"
            f"{train_result['group_accuracy']:.4f} | "
            f"val acc(12/5)={val_result['fine_accuracy']:.4f}/"
            f"{val_result['group_accuracy']:.4f} | "
            f"KL={train_result['kl']:.4f}"
        )

        if val_result["accuracy"] > best_tu_acc:
            best_tu_acc = val_result["accuracy"]
            best_tu_group_acc = val_result["group_accuracy"]
            save_phase_checkpoint(
                final_path,
                "tutoring",
                epoch,
                student,
                best_tu_acc,
                student.config,
                data,
                teacher=teacher,
                teacher_val_acc=cs_checkpoint.get("teacher_val_acc"),
            )

    final_checkpoint = load_checkpoint(final_path, map_location=device)
    student.load_state_dict(final_checkpoint["student_state_dict"])
    student.enable_quantization()

    test_result = evaluate_model(student, data["test_loader"], device)
    test_metrics = save_evaluation_outputs(
        test_result,
        OUTPUT_DIR,
        prefix="student_qkd_test",
    )
    stage_accuracy["tu"] = {
        "best_val_accuracy_12class": best_tu_acc,
        "best_val_accuracy_5class": best_tu_group_acc,
        "test_accuracy_12class": test_result["fine_accuracy"],
        "test_accuracy_5class": test_result["group_accuracy"],
    }
    print(
        f"[TU done] best val acc(12/5)={best_tu_acc:.4f}/"
        f"{best_tu_group_acc:.4f} | "
        f"test acc(12/5)={test_result['fine_accuracy']:.4f}/"
        f"{test_result['group_accuracy']:.4f}"
    )
    latency = measure_latency(student, data["test_dataset"], device)

    summary = {
        "method": "QKD: SS + CS + TU",
        "temperature": TEMPERATURE,
        "phase_epochs": {
            "ss": SS_EPOCHS,
            "cs": CS_EPOCHS,
            "tu": TU_EPOCHS,
        },
        "best_val_accuracy": {
            "ss": best_ss_acc,
            "cs": best_cs_acc,
            "tu": best_tu_acc,
        },
        "best_val_accuracy_5class": {
            "ss": best_ss_group_acc,
            "cs": best_cs_group_acc,
            "tu": best_tu_group_acc,
        },
        "stage_accuracy": stage_accuracy,
        "test": test_metrics,
        "student_parameters": student_params,
        "latency": latency,
        "checkpoint": str(final_path),
        "important_note": (
            "This checkpoint performs fake-quantized floating-point execution. "
            "A separate backend-specific conversion is required for real INT8 kernels."
        ),
    }

    (OUTPUT_DIR / "history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "quantizer_summary.json").write_text(
        json.dumps(student.quantizer_summary(), indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n==== QKD stage accuracy (12-class / 5-class) ====")
    print(
        f"{'stage':<6}{'best val 12':>13}{'best val 5':>12}"
        f"{'test 12':>10}{'test 5':>10}"
    )
    for stage_name in ("ss", "cs", "tu"):
        entry = stage_accuracy[stage_name]
        print(
            f"{stage_name.upper():<6}"
            f"{entry['best_val_accuracy_12class']:>13.4f}"
            f"{entry['best_val_accuracy_5class']:>12.4f}"
            f"{entry['test_accuracy_12class']:>10.4f}"
            f"{entry['test_accuracy_5class']:>10.4f}"
        )
    teacher_entry = stage_accuracy["cs"]
    print(
        f"{'CS-T':<6}"
        f"{teacher_entry['teacher_val_accuracy_12class_at_best_student']:>13.4f}"
        f"{teacher_entry['teacher_val_accuracy_5class_at_best_student']:>12.4f}"
        f"{teacher_entry['teacher_test_accuracy_12class']:>10.4f}"
        f"{teacher_entry['teacher_test_accuracy_5class']:>10.4f}"
    )
    print(
        "CS-T is the co-studied teacher; every other row is the student.\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
