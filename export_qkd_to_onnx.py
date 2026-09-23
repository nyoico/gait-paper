"""Export the student as a plain FP32 ONNX graph with fake quantization disabled.

The key difference from export_qkd_qat_qdq.py is the quantization preparation step.

    This file             : disable_quantization() -> FP32 graph without Q/DQ.
        For INT8, subsequently run quantize_onnx_int8.py to perform static
        quantization using training inputs for calibration. The scales learned
        by train_qkd.py are discarded and are not used in this export.

    export_qkd_qat_qdq.py  : Embed the learned scales directly in Q/DQ nodes.

The default output is deploy/student_qkd_fp32.onnx, which matches the default
input path of quantize_onnx_int8.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from qkd_model import QKDStudentModel, LearnedStepFakeQuantizer

@torch.no_grad()
def prepare_qat_model_for_onnx(model: QKDStudentModel) -> None:
    """
    Freeze the learned QAT scales and enable all fake quantizers.
    """

    uninitialized = []

    for name, module in model.named_modules():
        if not isinstance(module, LearnedStepFakeQuantizer):
            continue

        if not bool(module.initialized.item()):
            uninitialized.append(name)
            continue

        effective_scale = (
            module.scale.detach()
            .abs()
            .clamp_min(1e-8)
        )

        module.scale.copy_(effective_scale)
        module.scale.requires_grad_(False)
        module.enable()

    if uninitialized:
        raise RuntimeError(
            "Uninitialized quantizers were found: "
            + ", ".join(uninitialized)
        )

    model.eval()


DEFAULT_CLASSES = [
    "HC", "H_P", "H_C", "H_F",
    "K_P", "K_F", "K_R",
    "A_F", "A_R", "A_L",
    "C_F", "C_A",
]


class DeploymentWrapper(nn.Module):
    """ONNX interface: two GRF tensors in, one [B, C] logit tensor out."""

    def __init__(self, model: QKDStudentModel):
        super().__init__()
        self.model = model

    def forward(
        self,
        sensor_left: torch.Tensor,
        sensor_right: torch.Tensor,
    ) -> torch.Tensor:
        # zeros_like preserves the input batch dimension in the exported graph.
        tgt_seq = torch.zeros_like(
            sensor_left[:, :1, 0],
            dtype=torch.long,
        )
        seq_logits = self.model(
            sensor_left,
            sensor_right,
            tgt_seq,
            tgt_mask=None,
        )
        return seq_logits[-1]


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def save_metadata(checkpoint: dict[str, Any], output_dir: Path) -> None:
    normalization = checkpoint.get("normalization")
    if normalization is None:
        raise KeyError(
            "The checkpoint has no 'normalization' field. "
            "Save the training mean/std in the QKD checkpoint before export."
        )

    np.savez(
        output_dir / "normalization.npz",
        left_mean=np.asarray(normalization["left_mean"], dtype=np.float32),
        left_std=np.asarray(normalization["left_std"], dtype=np.float32),
        right_mean=np.asarray(normalization["right_mean"], dtype=np.float32),
        right_std=np.asarray(normalization["right_std"], dtype=np.float32),
    )

    label_to_idx = checkpoint.get(
        "label_to_idx",
        {label: index for index, label in enumerate(DEFAULT_CLASSES)},
    )
    idx_to_label = {
        str(index): label
        for label, index in sorted(label_to_idx.items(), key=lambda item: item[1])
    }
    (output_dir / "labels.json").write_text(
        json.dumps(idx_to_label, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def export_legacy(
    wrapper: nn.Module,
    left: torch.Tensor,
    right: torch.Tensor,
    output: Path,
    opset: int,
) -> None:
    torch.onnx.export(
        wrapper,
        (left, right),
        str(output),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["sensor_left", "sensor_right"],
        output_names=["logits"],
        dynamic_axes={
            "sensor_left": {0: "batch"},
            "sensor_right": {0: "batch"},
            "logits": {0: "batch"},
        },
        dynamo=False,
    )


def validate_onnx(path: Path) -> None:
    import onnx

    model = onnx.load(str(path))
    onnx.checker.check_model(model)


def compare_pytorch_and_onnx(
    wrapper: nn.Module,
    model_path: Path,
    left: torch.Tensor,
    right: torch.Tensor,
) -> None:
    import onnxruntime as ort

    with torch.no_grad():
        torch_logits = wrapper(left, right).cpu().numpy()

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    onnx_logits = session.run(
        ["logits"],
        {
            "sensor_left": left.cpu().numpy(),
            "sensor_right": right.cpu().numpy(),
        },
    )[0]

    max_abs_error = float(np.max(np.abs(torch_logits - onnx_logits)))
    same_prediction = bool(
        np.array_equal(
            np.argmax(torch_logits, axis=1),
            np.argmax(onnx_logits, axis=1),
        )
    )
    print(f"PyTorch/ONNX max |logit error|: {max_abs_error:.8f}")
    print(f"PyTorch/ONNX predictions equal: {same_prediction}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export best_student_qkd.pt to a plain FP32 ONNX deployment graph "
            "with fake quantization disabled. Run quantize_onnx_int8.py "
            "afterwards for static INT8, or use export_qkd_qat_qdq.py to keep "
            "the scales learned during QKD training."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/best_student_qkd.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("deploy/student_qkd_fp32.onnx"),
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--skip-runtime-check",
        action="store_true",
        help="Skip the PyTorch-versus-ONNX Runtime numerical comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint.get("student_config")
    state_dict = checkpoint.get("student_state_dict")
    if config is None or state_dict is None:
        raise KeyError(
            "Expected 'student_config' and 'student_state_dict' in the checkpoint."
        )

    model = QKDStudentModel(**config)
    model.load_state_dict(state_dict, strict=True)

    # The custom QAT fake quantizers are training-time simulations. Export a
    # clean graph and perform ONNX Runtime static INT8 conversion afterward.
    model.disable_quantization()
    model.eval()
    wrapper = DeploymentWrapper(model).eval()

    sensor_dim = int(config.get("sensor_dim", 5))
    max_len = int(config.get("max_len", 101))
    dummy_left = torch.zeros(1, sensor_dim, max_len, dtype=torch.float32)
    dummy_right = torch.zeros(1, sensor_dim, max_len, dtype=torch.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        export_legacy(
            wrapper,
            dummy_left,
            dummy_right,
            args.output,
            args.opset,
        )

    validate_onnx(args.output)
    save_metadata(checkpoint, args.output.parent)

    if not args.skip_runtime_check:
        compare_pytorch_and_onnx(
            wrapper,
            args.output,
            dummy_left,
            dummy_right,
        )

    print(f"FP32 ONNX: {args.output}")
    print(f"Normalization: {args.output.parent / 'normalization.npz'}")
    print(f"Labels: {args.output.parent / 'labels.json'}")


if __name__ == "__main__":
    main()
