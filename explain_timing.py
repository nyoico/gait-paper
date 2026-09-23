"""Run TIMING on the trained model from model.py / train.py.

Example:
    python explain_timing.py --max-samples 32 --n-steps 50

Set --max-samples 0 to explain the complete test set. Start with a small
number because attribution requires many forward/backward passes.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import Model
from plot_style import (
    DEFAULT_TIMING_SCHEMES,
    FIGURE_DPI,
    TIMING_SCHEME_NAMES,
    apply_paper_style,
    describe_timing_schemes,
    expand_scheme_names,
    resolve_timing_scheme,
)
from timing import (
    GRFClassifierWrapper,
    cumulative_probability_change,
    timing_attribute,
)


VALID_CLASSES = [
    "HC", "H_P", "H_C", "H_F",
    "K_P", "K_F", "K_R",
    "A_F", "A_R", "A_L",
    "C_F", "C_A",
]
COARSE_CLASSES = ["HC", "H", "K", "A", "C"]
FINE_TO_COARSE = {
    "HC": "HC",
    "H_P": "H", "H_C": "H", "H_F": "H",
    "K_P": "K", "K_F": "K", "K_R": "K",
    "A_F": "A", "A_R": "A", "A_L": "A",
    "C_F": "C", "C_A": "C",
}



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=Path("processed_data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/timing_12cls_ground_truth"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="0 means the complete test set")
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--num-segments", type=int, default=25)
    parser.add_argument("--min-seg-len", type=int, default=5)
    parser.add_argument("--max-seg-len", type=int, default=25)
    parser.add_argument("--alpha-batch-size", type=int, default=5)
    parser.add_argument("--metric-k", type=int, default=0,
                        help="0 disables CPD/CPP; otherwise remove this many cells")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-mode",
        choices=("predicted", "ground-truth"),
        default="ground-truth",
    )
    parser.add_argument(
        "--class-level",
        choices=("12", "5"),
        default="12",
        help="Explain the native 12-class output or grouped 5-class probability",
    )
    parser.add_argument(
        "--color-scheme",
        nargs="+",
        default=list(DEFAULT_TIMING_SCHEMES),
        metavar="NAME",
        choices=TIMING_SCHEME_NAMES + ["all", "cvd"],
        help=(
            "One or more color schemes, 'cvd' for every colorblind-safe "
            "scheme, or 'all'. Each scheme writes its own example heatmap "
            "files. The default writes "
            + ", ".join(DEFAULT_TIMING_SCHEMES)
            + " so that protanopia, deuteranopia and tritanopia readers each "
            "have a legible version. Available -- "
            + describe_timing_schemes()
        ),
    )
    parser.add_argument("--dpi", type=int, default=FIGURE_DPI)
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        help="Scale every font size relative to the publication defaults",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_channel_first(x: np.ndarray) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError(f"Expected a 3-D array, got {x.shape}")
    if x.shape[-1] == 5:
        x = np.transpose(x, (0, 2, 1))
    return x.astype(np.float32)


def normalize_from_training(
    train: np.ndarray,
    test: np.ndarray,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=(0, 2), keepdims=True)
    std = train.std(axis=(0, 2), keepdims=True)
    return (
        (train - mean) / (std + eps),
        (test - mean) / (std + eps),
        mean,
        std,
    )


def encode_labels(
    labels: np.ndarray,
    label_to_idx: dict[str, int],
    class_level: str,
) -> np.ndarray:
    if class_level == "12":
        return np.asarray([label_to_idx[str(x)] for x in labels], dtype=np.int64)

    coarse_to_idx = {label: idx for idx, label in enumerate(COARSE_CLASSES)}
    return np.asarray(
        [coarse_to_idx[FINE_TO_COARSE[str(x)]] for x in labels],
        dtype=np.int64,
    )


def make_coarse_groups(label_to_idx: dict[str, int]) -> list[list[int]]:
    return [
        [label_to_idx["HC"]],
        [label_to_idx["H_P"], label_to_idx["H_C"], label_to_idx["H_F"]],
        [label_to_idx["K_P"], label_to_idx["K_F"], label_to_idx["K_R"]],
        [label_to_idx["A_F"], label_to_idx["A_R"], label_to_idx["A_L"]],
        [label_to_idx["C_F"], label_to_idx["C_A"]],
    ]


def choose_initial_segments(
    time_steps: int,
    feature_dim: int,
    min_seg_len: int | None,
    max_seg_len: int | None,
    num_segments: int | None,
) -> tuple[int, int, int]:
    # These are practical starting values, not paper-mandated constants.
    min_len = max(1, round(0.05 * time_steps)) if min_seg_len is None else min_seg_len
    max_len = max(min_len, round(0.25 * time_steps)) if max_seg_len is None else max_seg_len
    max_len = min(max_len, time_steps)

    if num_segments is None:
        # Approximate 30% retained coverage, ignoring boundaries and overlap:
        # rho ~= 1-exp(-n*E[L]/(T*D)).
        expected_len = 0.5 * (min_len + max_len)
        target_fraction = 0.30
        n = math.ceil(
            -math.log(1.0 - target_fraction)
            * time_steps
            * feature_dim
            / expected_len
        )
        num_segments = max(1, n)

    return int(num_segments), int(min_len), int(max_len)


def build_model(checkpoint: dict, device: torch.device) -> tuple[Model, dict[str, int]]:
    config = checkpoint["config"]
    label_to_idx = checkpoint.get(
        "label_to_idx",
        {label: idx for idx, label in enumerate(VALID_CLASSES)},
    )
    model = Model(
        num_classes=len(label_to_idx),
        embed_dim=config["embed_dim"],
        sensor_dim=config["sensor_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        ff_dim=config["ff_dim"],
        dropout=config["dropout"],
        max_len=config["max_len"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Parameter gradients are unnecessary. Input gradients remain available.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model, label_to_idx


def save_example_heatmaps(
    signed_attr: np.ndarray,
    sensor_dim: int,
    output_dir: Path,
    color_schemes: list[str] | None = None,
    dpi: int = FIGURE_DPI,
) -> list[str]:
    """Save example heatmaps in multiple color schemes and return the schemes used.

    Multiple defaults accommodate differences in readability across CVD types.
    Colorbars are omitted; see example_heatmap_color_limits in metadata.json
    for the color ranges.
    """
    if signed_attr.shape[0] == 0:
        return []

    scheme_names = expand_scheme_names(color_schemes)

    left = signed_attr[0, :, :sensor_dim].T
    right = signed_attr[0, :, sensor_dim:].T

    left_limit = max(float(np.abs(left).max()), 1e-12)
    right_limit = max(float(np.abs(right).max()), 1e-12)

    for scheme_name in scheme_names:
        scheme = resolve_timing_scheme(scheme_name)
        for values, limit, side, stem in (
            (left, left_limit, "Left", "sample_000_left_signed"),
            (right, right_limit, "Right", "sample_000_right_signed"),
        ):
            plt.figure(figsize=(13, 5))
            plt.imshow(
                values,
                aspect="auto",
                origin="lower",
                vmin=-limit,
                vmax=limit,
                cmap=scheme.signed,
            )
            plt.xlabel("Time index")
            plt.ylabel(f"{side} sensor channel")
            plt.title(f"TIMING signed attribution: {side.lower()} sensor")
            plt.tight_layout()
            plt.savefig(output_dir / f"{stem}_{scheme_name}.png", dpi=dpi)
            plt.close()

    return scheme_names


def main() -> None:
    args = parse_args()
    apply_paper_style(scale=args.font_scale)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model, label_to_idx = build_model(checkpoint, device)
    sensor_dim = int(checkpoint["config"]["sensor_dim"])

    left_train = ensure_channel_first(
        np.load(args.processed_dir / "X_left_train.npy").astype(np.float32)
    )
    left_test = ensure_channel_first(
        np.load(args.processed_dir / "X_left_test.npy").astype(np.float32)
    )
    right_train = ensure_channel_first(
        np.load(args.processed_dir / "X_right_train.npy").astype(np.float32)
    )
    right_test = ensure_channel_first(
        np.load(args.processed_dir / "X_right_test.npy").astype(np.float32)
    )
    y_left_test_raw = np.load(
        args.processed_dir / "y_left_test.npy", allow_pickle=True
    )
    y_right_test_raw = np.load(
        args.processed_dir / "y_right_test.npy", allow_pickle=True
    )

    _, left_test, left_mean, left_std = normalize_from_training(left_train, left_test)
    _, right_test, right_mean, right_std = normalize_from_training(right_train, right_test)

    if left_test.shape != right_test.shape:
        raise ValueError(
            f"Left/right test arrays differ: {left_test.shape} vs {right_test.shape}"
        )
    if left_test.shape[1] != sensor_dim:
        raise ValueError(
            f"Checkpoint sensor_dim={sensor_dim}, but data has {left_test.shape[1]} channels"
        )

    if not np.array_equal(y_left_test_raw.astype(str), y_right_test_raw.astype(str)):
        print(
            "WARNING: y_left_test and y_right_test are not identical. "
            "The training code uses y_left_test, so this script does the same."
        )

    labels = encode_labels(y_left_test_raw, label_to_idx, args.class_level)
    # [N, C, T] -> [N, T, 2C]
    packed = np.concatenate(
        [left_test.transpose(0, 2, 1), right_test.transpose(0, 2, 1)],
        axis=-1,
    ).astype(np.float32)

    if args.max_samples > 0:
        packed = packed[: args.max_samples]
        labels = labels[: args.max_samples]

    dataset = TensorDataset(
        torch.from_numpy(packed),
        torch.from_numpy(labels),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    class_groups = make_coarse_groups(label_to_idx) if args.class_level == "5" else None
    wrapper = GRFClassifierWrapper(
        model,
        sensor_dim=sensor_dim,
        return_probabilities=True,
        class_groups=class_groups,
    ).to(device)
    wrapper.eval()

    time_steps = packed.shape[1]
    feature_dim = packed.shape[2]
    num_segments, min_seg_len, max_seg_len = choose_initial_segments(
        time_steps=time_steps,
        feature_dim=feature_dim,
        min_seg_len=args.min_seg_len,
        max_seg_len=args.max_seg_len,
        num_segments=args.num_segments,
    )

    all_attr: list[torch.Tensor] = []
    all_counts: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_cpd: list[torch.Tensor] = []
    all_cpp: list[torch.Tensor] = []

    for batch_index, (x_btd, y_true) in enumerate(loader):
        x_btd = x_btd.to(device)
        y_true = y_true.to(device)
        baseline = torch.zeros_like(x_btd)

        with torch.no_grad():
            original_probs = wrapper(x_btd)
            predicted = original_probs.argmax(dim=-1)
        targets = predicted if args.target_mode == "predicted" else y_true

        result = timing_attribute(
            wrapper,
            x_btd,
            baseline=baseline,
            targets=targets,
            n_steps=args.n_steps,
            num_segments=num_segments,
            min_seg_len=min_seg_len,
            max_seg_len=max_seg_len,
            alpha_batch_size=args.alpha_batch_size,
            seed=args.seed + batch_index,
        )

        all_attr.append(result.attribution.cpu())
        all_counts.append(result.interpolated_count.cpu())
        all_targets.append(result.targets.cpu())
        all_probs.append(result.original_output.cpu())
        all_labels.append(y_true.cpu())

        if args.metric_k > 0:
            k = min(args.metric_k, x_btd[0].numel())
            all_cpd.append(
                cumulative_probability_change(
                    wrapper,
                    x_btd,
                    result.attribution,
                    baseline=baseline,
                    k=k,
                    remove_largest_first=True,
                ).cpu()
            )
            all_cpp.append(
                cumulative_probability_change(
                    wrapper,
                    x_btd,
                    result.attribution,
                    baseline=baseline,
                    k=k,
                    remove_largest_first=False,
                ).cpu()
            )

        print(
            f"Explained {min((batch_index + 1) * args.batch_size, len(dataset))}"
            f"/{len(dataset)} samples"
        )

    signed = torch.cat(all_attr).numpy()
    counts = torch.cat(all_counts).numpy()
    targets = torch.cat(all_targets).numpy()
    probabilities = torch.cat(all_probs).numpy()
    true_labels = torch.cat(all_labels).numpy()

    np.save(args.output_dir / "timing_signed.npy", signed)
    absolute = np.abs(signed)
    np.save(args.output_dir / "timing_abs.npy", absolute)
    np.save(args.output_dir / "timing_positive.npy", np.clip(signed, 0.0, None))
    np.save(args.output_dir / "timing_negative_magnitude.npy", np.clip(-signed, 0.0, None))
    np.save(args.output_dir / "time_importance_abs.npy", absolute.mean(axis=-1))
    np.save(args.output_dir / "feature_importance_abs.npy", absolute.mean(axis=1))
    np.save(args.output_dir / "timing_left_signed.npy", signed[:, :, :sensor_dim])
    np.save(args.output_dir / "timing_right_signed.npy", signed[:, :, sensor_dim:])
    np.save(args.output_dir / "interpolated_count.npy", counts)
    np.save(args.output_dir / "targets.npy", targets)
    np.save(args.output_dir / "true_labels.npy", true_labels)
    np.save(args.output_dir / "original_probabilities.npy", probabilities)
    np.savez(
        args.output_dir / "normalization_stats.npz",
        left_mean=left_mean,
        left_std=left_std,
        right_mean=right_mean,
        right_std=right_std,
    )

    metrics: dict[str, float | int] = {}
    if all_cpd:
        cpd = torch.cat(all_cpd).numpy()
        cpp = torch.cat(all_cpp).numpy()
        np.save(args.output_dir / "cpd.npy", cpd)
        np.save(args.output_dir / "cpp.npy", cpp)
        metrics = {
            "metric_k": int(min(args.metric_k, time_steps * feature_dim)),
            "cpd_mean": float(cpd.mean()),
            "cpp_mean": float(cpp.mean()),
        }

    metadata = {
        "device": str(device),
        "num_samples": int(len(dataset)),
        "input_shape": [int(v) for v in packed.shape],
        "sensor_dim_per_side": sensor_dim,
        "target_mode": args.target_mode,
        "class_level": args.class_level,
        "output_classes": COARSE_CLASSES if args.class_level == "5" else VALID_CLASSES,
        "baseline": "zero in standardized space (training-channel mean)",
        "n_steps": args.n_steps,
        "num_segments": num_segments,
        "min_seg_len": min_seg_len,
        "max_seg_len": max_seg_len,
        "alpha_batch_size": args.alpha_batch_size,
        "seed": args.seed,
        "prediction_accuracy_on_explained_subset": float(
            (probabilities.argmax(axis=1) == true_labels).mean()
        ),
        **metrics,
    }
    written_schemes = save_example_heatmaps(
        signed,
        sensor_dim,
        args.output_dir,
        color_schemes=args.color_scheme,
        dpi=args.dpi,
    )
    print(f"example heatmap color schemes: {', '.join(written_schemes) or '(none)'}")

    # Record example heatmap color limits in metadata because colorbars are omitted.
    if signed.shape[0] > 0:
        left_limit = max(float(np.abs(signed[0, :, :sensor_dim]).max()), 1e-12)
        right_limit = max(float(np.abs(signed[0, :, sensor_dim:]).max()), 1e-12)
        metadata["example_heatmap_color_limits"] = {
            "left_signed_symmetric_limit": left_limit,
            "right_signed_symmetric_limit": right_limit,
            "note": (
                "Figures carry no colorbar. Each signed heatmap spans "
                "[-limit, +limit] of its own side."
            ),
        }
    metadata["example_heatmap_color_schemes"] = written_schemes

    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
