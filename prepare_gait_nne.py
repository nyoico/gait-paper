#!/usr/bin/env python3
"""Prepare the supplied gait Model/train.py pipeline for Unreal NNE.

No training, no changes to model.py/train.py/checkpoints/processed_data.
Only use YOUR trusted model.py and checkpoint. Model code is imported and executed.
The checkpoint loader uses weights_only=True and has no unsafe automatic fallback.

Input ONNX tensors are the values stored in processed_data BEFORE the z-score in
train.py. They are NOT necessarily raw device values or physical units. Channel
order is preserved, not guessed. Z-score statistics are embedded into the graph.

First run (from the training project, with this script alongside model.py):
    python prepare_gait_nne.py
Dependencies: existing PyTorch + NumPy; install onnx and onnxruntime separately.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import platform
import sys
import traceback
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

try:
    import numpy as np
    import torch
    from torch import nn
except ImportError as exc:
    raise SystemExit(
        f"Missing {exc.name}. Run this script with the Python environment that "
        "already runs your training code. Do not reinstall PyTorch blindly."
    ) from exc

EPS = 1e-6
EXPECTED_LABELS = {
    "HC", "H_P", "H_C", "H_F", "K_P", "K_F", "K_R",
    "A_F", "A_R", "A_L", "C_F", "C_A",
}
MODEL_CONFIG_KEYS = (
    "embed_dim", "sensor_dim", "num_heads", "num_layers",
    "ff_dim", "dropout", "max_len",
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_files(paths: list[Path]) -> None:
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Required files were not found:\n  " + "\n  ".join(missing))


def channel_first(x: np.ndarray, label: str, time_steps: int) -> np.ndarray:
    # Same rule and float32 conversion as the supplied ensure_channel_first().
    if x.ndim != 3:
        raise ValueError(f"{label}: expected a 3D array, got {x.shape}")
    if x.shape[-1] == 5:
        x = np.transpose(x, (0, 2, 1))
    x = x.astype(np.float32)
    if x.shape[0] < 1 or tuple(x.shape[1:]) != (5, time_steps):
        raise ValueError(
            f"{label}: expected [N,5,{time_steps}] after the train.py transpose, "
            f"got {x.shape}. No padding, truncation or channel reordering is performed."
        )
    # Bound temporary memory used for validation on large datasets.
    for start in range(0, len(x), 512):
        if not np.isfinite(x[start:start + 512]).all():
            raise ValueError(f"{label}: contains NaN/Inf")
    return x


def load_array(path: Path, time_steps: int) -> np.ndarray:
    # Match load_data() followed by ensure_channel_first(), including dtype/order.
    x = np.load(path, allow_pickle=False).astype(np.float32)
    return channel_first(x, path.name, time_steps)


def training_statistics(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Must use the original full X_*_train.npy, not test/one sample/per-window.
    # This preserves the supplied train.py policy: statistics BEFORE random_split.
    mean = x.mean(axis=(0, 2), keepdims=True)
    std = x.std(axis=(0, 2), keepdims=True)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Non-finite normalization statistics")
    if (std == 0).any():
        print("[WARN] A training channel has zero std; preserving std + 1e-6.")
    return mean, std


def load_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            "Checkpoint safe loading failed. Use your own best_model.pt from the "
            "supplied train.py. This script does not retry with weights_only=False. "
            f"Original error: {exc}"
        ) from exc
    if not isinstance(ckpt, dict):
        raise ValueError("Expected a checkpoint dictionary, not a saved nn.Module")
    for key in ("model_state_dict", "config"):
        if key not in ckpt:
            raise ValueError(f"Checkpoint is missing {key!r}; use best_model.pt")
    missing = set(MODEL_CONFIG_KEYS) - set(ckpt["config"])
    if missing:
        raise ValueError(f"Missing checkpoint config fields: {sorted(missing)}")
    cfg = {k: ckpt["config"][k] for k in MODEL_CONFIG_KEYS}
    for key in MODEL_CONFIG_KEYS:
        cfg[key] = float(cfg[key]) if key == "dropout" else int(cfg[key])
    if cfg["sensor_dim"] != 5:
        raise ValueError(f"Expected 5 channels per side; got {cfg['sensor_dim']}")
    if cfg["max_len"] < 2 or cfg["embed_dim"] % cfg["num_heads"]:
        raise ValueError("Invalid time length or attention dimensions in config")
    state = ckpt["model_state_dict"]
    if "classifier.weight" not in state:
        raise ValueError("classifier.weight is missing from model_state_dict")
    num_classes = int(state["classifier.weight"].shape[0])
    if "idx_to_label" in ckpt:
        mapping = {int(k): str(v) for k, v in ckpt["idx_to_label"].items()}
    elif "label_to_idx" in ckpt:
        mapping = {int(v): str(k) for k, v in ckpt["label_to_idx"].items()}
    else:
        raise ValueError("Checkpoint has no label mapping; class order cannot be guessed")
    if set(mapping) != set(range(num_classes)):
        raise ValueError("Label indices do not match classifier output dimensions")
    labels = [mapping[i] for i in range(num_classes)]
    if num_classes != 12 or set(labels) != EXPECTED_LABELS:
        raise ValueError("This exporter expects the supplied 12-class gait model")
    if "label_to_idx" in ckpt:
        inverse = {str(k): int(v) for k, v in ckpt["label_to_idx"].items()}
        if inverse != {name: i for i, name in enumerate(labels)}:
            raise ValueError("Checkpoint label mappings disagree")
    return ckpt, cfg, labels


def instantiate_model(model_path: Path, cfg: dict[str, Any], state: dict,
                      num_classes: int) -> nn.Module:
    spec = importlib.util.spec_from_file_location("_gait_nne_user_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {model_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "Model"):
        raise ValueError("model.py must define the supplied Model class")
    model = mod.Model(num_classes=num_classes, **cfg).cpu().float()
    model.load_state_dict(state, strict=True)
    model.eval()
    model.requires_grad_(False)
    return model


class InferenceWrapper(nn.Module):
    """Fixed batch=1 wrapper. Normalization + zero decoder token + last logits."""
    def __init__(self, model: nn.Module, mean_l: np.ndarray, std_l: np.ndarray,
                 mean_r: np.ndarray, std_r: np.ndarray):
        super().__init__()
        self.model = model
        for name, a in (("mean_l", mean_l), ("denom_l", std_l + EPS),
                        ("mean_r", mean_r), ("denom_r", std_r + EPS)):
            self.register_buffer(name, torch.from_numpy(a.copy()).float())
        self.register_buffer("tgt_seq", torch.zeros((1, 1), dtype=torch.long))
        # make_casual_mask(1) in train.py is this all-False mask.
        self.register_buffer("tgt_mask", torch.zeros((1, 1), dtype=torch.bool))

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left_z = (left - self.mean_l) / self.denom_l
        right_z = (right - self.mean_r) / self.denom_r
        return self.model(left_z, right_z, self.tgt_seq,
                          tgt_mask=self.tgt_mask)[-1]


@torch.no_grad()
def reference_logits(model: nn.Module, left: np.ndarray, right: np.ndarray,
                     ml: np.ndarray, sl: np.ndarray,
                     mr: np.ndarray, sr: np.ndarray) -> np.ndarray:
    # Independent NumPy z-score path, matching train.py rather than the wrapper.
    lz = np.ascontiguousarray((left - ml) / (sl + EPS))
    rz = np.ascontiguousarray((right - mr) / (sr + EPS))
    tgt = torch.zeros((1, 1), dtype=torch.long)
    mask = torch.triu(torch.ones(1, 1, dtype=torch.bool), diagonal=1)
    logits = model(torch.from_numpy(lz), torch.from_numpy(rz), tgt,
                   tgt_mask=mask)[-1]
    return logits.cpu().numpy()


def comparison(actual: np.ndarray, expected: np.ndarray,
               atol: float, rtol: float) -> dict[str, Any]:
    if actual.shape != expected.shape:
        return {"allclose": False, "shape_error": [list(actual.shape), list(expected.shape)]}
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        return {"allclose": False, "nonfinite": True}
    return {
        "allclose": bool(np.allclose(actual, expected, atol=atol, rtol=rtol)),
        "max_abs_error": float(np.max(np.abs(actual - expected))),
        "argmax_match": bool(np.array_equal(actual.argmax(axis=-1), expected.argmax(axis=-1))),
    }


def validate_one(c: dict[str, Any]) -> bool:
    return bool(c.get("allclose", False) and c.get("argmax_match", False))


def softmax(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    ex = np.exp(a - a.max(axis=-1, keepdims=True))
    return ex / ex.sum(axis=-1, keepdims=True)


def export_sample_csv(path: Path, left: np.ndarray, right: np.ndarray) -> None:
    # Generic channel IDs intentionally avoid claiming unverified channel semantics.
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Name", "FrameIndex"] + [f"L_C{i}" for i in range(5)]
                   + [f"R_C{i}" for i in range(5)])
        for t in range(left.shape[-1]):
            vals = [*left[0, :, t], *right[0, :, t]]
            # 9 significant digits round-trip float32 values.
            w.writerow([f"F{t:04d}", t] + [format(float(v), ".9g") for v in vals])


def prepare(args: argparse.Namespace, out: Path) -> None:
    root = args.project_root.resolve()
    def resolve_arg(p: Path) -> Path:
        return (p if p.is_absolute() else root / p).resolve()
    model_path = resolve_arg(args.model_file)
    checkpoint_path = resolve_arg(args.checkpoint)
    data_dir = resolve_arg(args.data_dir)
    files = {side + "_" + split: data_dir / f"X_{side}_{split}.npy"
             for side in ("left", "right") for split in ("train", "test")}
    require_files([model_path, checkpoint_path, *files.values()])
    print("[1/6] Loading checkpoint and model configuration...")
    ckpt, cfg, labels = load_checkpoint(checkpoint_path)
    steps = cfg["max_len"]
    print(f"      Inputs: left=[1,5,{steps}], right=[1,5,{steps}], float32")
    print(f"      Output: logits=[1,{len(labels)}]")
    model = instantiate_model(model_path, cfg, ckpt["model_state_dict"], len(labels))
    # Release optimizer state carried by the original checkpoint.
    del ckpt

    print("[2/6] Reproducing train.py normalization (full original train arrays)...")
    left_train = load_array(files["left_train"], steps)
    train_count_l = len(left_train)
    mean_l, std_l = training_statistics(left_train)
    del left_train
    right_train = load_array(files["right_train"], steps)
    train_count_r = len(right_train)
    mean_r, std_r = training_statistics(right_train)
    del right_train
    if train_count_l != train_count_r:
        raise ValueError("Left/right training sample counts differ")
    left_test = load_array(files["left_test"], steps)
    right_test = load_array(files["right_test"], steps)
    if len(left_test) != len(right_test):
        raise ValueError("Left/right test sample counts differ")
    if not 0 <= args.sample_index < len(left_test):
        raise ValueError(f"sample-index must be in [0,{len(left_test)-1}]")
    indices = np.unique(np.r_[args.sample_index, np.linspace(
        0, len(left_test) - 1, min(args.verify_samples, len(left_test)), dtype=int)]).tolist()
    wrapper = InferenceWrapper(model, mean_l, std_l, mean_r, std_r).eval()

    stats = {"source": "Full original train arrays, before validation random_split",
             "formula": "(x-mean)/(std+1e-6)", "eps": EPS, "ddof": 0,
             "statistics_dtype": "float32", "embedded_in_onnx": True,
             "left": {"mean": mean_l.reshape(-1).tolist(), "std": std_l.reshape(-1).tolist()},
             "right": {"mean": mean_r.reshape(-1).tolist(), "std": std_r.reshape(-1).tolist()}}
    write_json(out / "normalization.json", stats)
    print("[3/6] Checking wrapped PyTorch inference against train.py-style inference...")
    records = []
    refs: dict[int, np.ndarray] = {}
    for idx in indices:
        l = np.ascontiguousarray(left_test[idx:idx+1])
        r = np.ascontiguousarray(right_test[idx:idx+1])
        ref = reference_logits(model, l, r, mean_l, std_l, mean_r, std_r)
        if tuple(ref.shape) != (1, len(labels)):
            raise ValueError(f"Unexpected reference output shape: {ref.shape}")
        with torch.no_grad():
            got = wrapper(torch.from_numpy(l), torch.from_numpy(r)).numpy()
        check = comparison(got, ref, args.atol, args.rtol)
        records.append({"sample_index": idx, "wrapper_vs_reference": check})
        refs[idx] = ref
        if not validate_one(check):
            write_json(out / "validation_report.json", {"status": "FAIL_WRAPPER", "samples": records})
            raise RuntimeError("Wrapped inference differs from the supplied training inference")

    try:
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            f"Missing {exc.name}. Install onnx and onnxruntime in the chosen export "
            "environment, then rerun with a new --output directory."
        ) from exc
    print("[4/6] Exporting fixed-shape ONNX (opset 17)...")
    pending = out / "gait_classifier.UNVERIFIED.onnx"
    i = args.sample_index
    sample_l = np.ascontiguousarray(left_test[i:i+1])
    sample_r = np.ascontiguousarray(right_test[i:i+1])
    kwargs = dict(export_params=True, opset_version=17,
                  input_names=["left", "right"], output_names=["logits"],
                  keep_initializers_as_inputs=False, dynamic_axes=None)
    sig = inspect.signature(torch.onnx.export)
    if "dynamo" in sig.parameters:
        # Explicit legacy path for this fixed-shape, opset-17 deployment trial.
        kwargs["dynamo"] = False
    if "external_data" in sig.parameters:
        kwargs["external_data"] = False
    mha = getattr(torch.backends, "mha", None)
    old_fastpath = mha.get_fastpath_enabled() if mha else None
    try:
        if mha:
            mha.set_fastpath_enabled(False)
        with torch.no_grad():
            torch.onnx.export(wrapper, (torch.from_numpy(sample_l), torch.from_numpy(sample_r)),
                              str(pending), **kwargs)
    finally:
        if mha:
            mha.set_fastpath_enabled(old_fastpath)
    graph = onnx.load(str(pending))
    onnx.checker.check_model(graph, full_check=True)
    if any(t.data_location == onnx.TensorProto.EXTERNAL for t in graph.graph.initializer):
        raise RuntimeError("Unexpected external weights; do not import this model yet")
    ir_version = int(graph.ir_version)
    opset_versions = {o.domain or "ai.onnx": int(o.version) for o in graph.opset_import}
    del graph

    print("[5/6] Checking ONNX Runtime CPU against original PyTorch logits...")
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(str(pending), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    actual_inputs = session.get_inputs()
    if [a.name for a in actual_inputs] != ["left", "right"]:
        raise ValueError("ONNX input names/order changed unexpectedly")
    if any(a.type != "tensor(float)" or a.shape != [1, 5, steps] for a in actual_inputs):
        raise ValueError("ONNX input dtype/shape does not match the deployment contract")
    if session.get_outputs()[0].name != "logits":
        raise ValueError("Unexpected ONNX output name")
    for rec in records:
        idx = rec["sample_index"]
        output = session.run(["logits"], {
            "left": np.ascontiguousarray(left_test[idx:idx+1]),
            "right": np.ascontiguousarray(right_test[idx:idx+1]),
        })[0]
        rec["onnx_vs_reference"] = comparison(output, refs[idx], args.atol, args.rtol)
    passed = all(validate_one(r["onnx_vs_reference"]) for r in records)
    versions = {"python": platform.python_version(), "platform": platform.platform(),
                "torch": str(torch.__version__), "numpy": np.__version__,
                "onnx": onnx.__version__, "onnxruntime": ort.__version__}
    report = {"status": "PASS_PYTORCH_ONNX" if passed else "FAIL_ONNX_PARITY",
              "unreal_runtime_status": "NOT_TESTED",
              "tested_samples": len(records), "atol": args.atol, "rtol": args.rtol,
              "max_abs_logit_error": max(
                  r["onnx_vs_reference"].get("max_abs_error", 0.0) for r in records),
              "all_argmax_match": all(r["onnx_vs_reference"].get("argmax_match", False) for r in records),
              "versions": versions, "samples": records,
              "scope": "Numerical conversion check, not classification accuracy or clinical validation"}
    write_json(out / "validation_report.json", report)
    if not passed:
        raise RuntimeError("ONNX parity failed. Do not import the UNVERIFIED ONNX into Unreal.")
    del session

    print("[6/6] Writing one sample and the deployment contract...")
    export_sample_csv(out / "sample_frames.csv", sample_l, sample_r)
    write_json(out / "sample_inputs.json", {
        "sample_index": i, "shape_per_side": [1, 5, steps],
        "flattening": "channel-major: offset = channel_index * T + frame_index",
        "input_stage": "processed_npy_before_train_py_zscore",
        "left": sample_l.reshape(-1).tolist(), "right": sample_r.reshape(-1).tolist(),
    })
    logits = refs[i]
    probs = softmax(logits)
    pred = int(logits.argmax(axis=-1)[0])
    write_json(out / "reference_prediction.json", {
        "sample_index": i, "source": "original PyTorch reference, NOT Unreal inference",
        "labels": labels, "logits": logits[0].tolist(), "probabilities": probs[0].tolist(),
        "predicted_index": pred, "predicted_label": labels[pred],
        "true_label": None, "true_label_status": "not_loaded",
    })
    manifest = {
        "schema_version": 1, "model_config": cfg,
        "model_filename": "gait_classifier.onnx", "onnx_ir_version": ir_version,
        "opsets": opset_versions,
        "inputs": [{"name": name, "dtype": "float32", "shape": [1, 5, steps]}
                   for name in ("left", "right")],
        "outputs": [{"name": "logits", "dtype": "float32", "shape": [1, 12]}],
        "labels": labels, "normalization_embedded": True,
        "input_stage": "processed_npy_before_train_py_zscore",
        "do_not_zscore_in_unreal": True,
        "upstream_preprocessing": "Must match how original processed_data arrays were produced; not reconstructed here",
        "channel_order": "Original stored array order unchanged; C0..C4 are zero-based indices",
        "channel_semantics_verified": False, "physical_units_verified": False,
        "left_right_clock_alignment_verified": False,
        "flattening": "channel-major: offset = channel_index * T + frame_index",
        "sequence_policy": "One complete preprocessed segment per inference, not one force frame",
        "suggested_first_runtime": "NNERuntimeORTCpu",
        "unreal_runtime_status": "NOT_TESTED",
        "sample_index": i,
        "files_sha256": {"model.py": sha256(model_path), "checkpoint": sha256(checkpoint_path),
                         **{key: sha256(path) for key, path in files.items()}},
        "normalization_source_warning": "Hashes record files used NOW; they do not prove they match files used during training",
        "versions": versions,
    }
    write_json(out / "deployment.json", manifest)
    # Only a numerically verified export receives the final deployable filename.
    pending.rename(out / "gait_classifier.onnx")
    print(f"[PASS] PyTorch / ONNX parity on {len(records)} sample(s)")
    print(f"       Max abs logit error: {report['max_abs_logit_error']:.8g}")
    print(f"       Reference sample {i}: {labels[pred]} (not a correctness claim)")
    print(f"       Output folder: {out}")
    print("[NEXT] Import gait_classifier.onnx, then verify inference INSIDE Unreal.")
    print("[NOTE] Unreal runtime execution has NOT been tested by this script.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--model-file", type=Path, default=Path("model.py"))
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_model.pt"))
    p.add_argument("--data-dir", type=Path, default=Path("processed_data"))
    p.add_argument("--output", type=Path, default=Path("ue_export"),
                   help="NEW output directory; an existing directory will never be overwritten")
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--verify-samples", type=int, default=16)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--rtol", type=float, default=1e-4)
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.verify_samples < 1 or args.threads < 1 or args.atol <= 0 or args.rtol < 0:
        print("[FAIL] Invalid verification settings", file=sys.stderr)
        return 2
    root = args.project_root.resolve()
    out = (args.output if args.output.is_absolute() else root / args.output).resolve()
    if out.exists():
        print(f"[FAIL] Output already exists: {out}\n"
              "No files were changed. Use --output ue_export_v2 for another attempt.", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=False)
    try:
        torch.set_num_threads(args.threads)
        prepare(args, out)
        return 0
    except Exception as exc:
        error = traceback.format_exc()
        (out / "error.txt").write_text(error, encoding="utf-8")
        print(f"[FAIL] {exc}\nFull traceback: {out / 'error.txt'}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
