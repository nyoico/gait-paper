import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from qkd_common import (
    LABEL_TO_IDX,
    load_checkpoint,
    VALID_CLASSES,
    count_parameters,
    evaluate_model,
    extract_last_logits,
    fine_to_group_indices,
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
OUTPUT_DIR = Path("output/student_fp")

BATCH_SIZE = 32
NUM_WORKERS = 0
VAL_RATIO = 0.03
EPOCHS = 300
PATIENCE = 20
LR = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05

STUDENT_CONFIG = {
    "num_classes": len(VALID_CLASSES),
    "embed_dim": 192,
    "sensor_dim": 5,
    "num_heads": 6,
    "num_layers": 4,
    "ff_dim": 768,
    "dropout": 0.1,
    "max_len": 101,
    "weight_bits": 8,
    "activation_bits": 8,
    "first_last_bits": 8,
}


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_true = []
    all_pred = []

    for left, right, labels in loader:
        left = left.to(device, non_blocking=True)
        right = right.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        tgt_seq, tgt_mask = make_tgt_inputs(labels.size(0), device)

        logits = extract_last_logits(
            model(left, right, tgt_seq, tgt_mask=tgt_mask)
        )
        loss = F.cross_entropy(
            logits,
            labels,
            label_smoothing=LABEL_SMOOTHING,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)
        all_true.extend(labels.detach().cpu().tolist())
        all_pred.extend(preds.detach().cpu().tolist())

    y_true = torch.tensor(all_true, dtype=torch.long).numpy()
    y_pred = torch.tensor(all_pred, dtype=torch.long).numpy()
    group_true = fine_to_group_indices(y_true)
    group_pred = fine_to_group_indices(y_pred)

    fine_accuracy = total_correct / max(total_count, 1)
    group_accuracy = (
        float((group_true == group_pred).mean()) if len(group_true) else 0.0
    )

    return {
        "loss": total_loss / max(total_count, 1),
        "accuracy": fine_accuracy,
        "fine_accuracy": fine_accuracy,
        "group_accuracy": group_accuracy,
        "accuracy_12class": fine_accuracy,
        "accuracy_5class": group_accuracy,
    }


def main():
    set_seed(SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = prepare_data(
        processed_dir=PROCESSED_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        val_ratio=VAL_RATIO,
        seed=SEED,
    )

    config = dict(STUDENT_CONFIG)
    config["sensor_dim"] = data["sensor_dim"]
    config["max_len"] = data["max_len"]

    model = QKDStudentModel(**config).to(device)
    model.disable_quantization()

    params = count_parameters(model)
    print(json.dumps(params, indent=2))
    if params["total"] > 5_000_000:
        raise RuntimeError(f"Student exceeds 5M parameters: {params['total']:,}")

    # Quantizer scale parameters are inactive in FP training and are excluded.
    fp_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if ".scale" not in name
    ]
    optimizer = torch.optim.AdamW(
        fp_parameters,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    history = []
    best_val_acc = -1.0
    best_epoch = 0
    patience_count = 0
    best_path = CHECKPOINT_DIR / "best_student_fp.pt"

    for epoch in tqdm(range(1, EPOCHS + 1), desc="FP student"):
        train_result = train_epoch(
            model,
            data["train_loader"],
            optimizer,
            device,
        )
        val_result = evaluate_model(model, data["val_loader"], device)
        # Model selection remains based on the original 12-class task.
        scheduler.step(val_result["fine_accuracy"])

        row = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "train_accuracy_12class": train_result["fine_accuracy"],
            "train_accuracy_5class": train_result["group_accuracy"],
            "val_loss": val_result["loss"],
            "val_accuracy_12class": val_result["fine_accuracy"],
            "val_accuracy_5class": val_result["group_accuracy"],
            # Legacy aliases retained for scripts that expect these keys.
            "train_accuracy": train_result["fine_accuracy"],
            "val_accuracy": val_result["fine_accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        tqdm.write(
            f"[FP {epoch:03d}] "
            f"train acc(12/5)={train_result['fine_accuracy']:.4f}/"
            f"{train_result['group_accuracy']:.4f} | "
            f"val acc(12/5)={val_result['fine_accuracy']:.4f}/"
            f"{val_result['group_accuracy']:.4f} | "
            f"val loss={val_result['loss']:.4f}"
        )

        if val_result["fine_accuracy"] > best_val_acc:
            best_val_acc = val_result["fine_accuracy"]
            best_epoch = epoch
            patience_count = 0
            torch.save(
                {
                    "stage": "fp_student",
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "config": config,
                    "label_to_idx": LABEL_TO_IDX,
                    "normalization": data["normalization"],
                    "split_indices": data["split_indices"],
                    "parameter_count": params,
                    "quantization_enabled": False,
                },
                best_path,
            )
        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}")
            break

    checkpoint = load_checkpoint(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.disable_quantization()

    test_result = evaluate_model(model, data["test_loader"], device)
    metrics = save_evaluation_outputs(
        test_result,
        OUTPUT_DIR,
        prefix="student_fp_test",
    )
    latency = measure_latency(model, data["test_dataset"], device)

    summary = {
        "best_epoch": best_epoch,
        "best_val_accuracy_12class": best_val_acc,
        "best_val_accuracy": best_val_acc,  # backward-compatible alias
        "test_accuracy_12class": metrics["accuracy_12class"],
        "test_accuracy_5class": metrics["accuracy_5class"],
        "test": metrics,
        "parameters": params,
        "latency": latency,
        "checkpoint": str(best_path),
    }
    (OUTPUT_DIR / "history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(
        f"Test accuracy | 12-class: {metrics['accuracy_12class']:.4f} "
        f"({metrics['accuracy_12class'] * 100:.2f}%) | "
        f"5-class: {metrics['accuracy_5class']:.4f} "
        f"({metrics['accuracy_5class'] * 100:.2f}%)"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
