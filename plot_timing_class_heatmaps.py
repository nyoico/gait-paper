"""Create class-wise TIMING heatmaps from explain_timing.py outputs.

Recommended usage for all 12 classes:

1) Generate attribution for the complete test set with each sample's true class
   as the attribution target:

   python explain_timing.py \
       --max-samples 0 \
       --class-level 12 \
       --target-mode ground-truth \
       --output-dir output/timing_12cls_true \
       --n-steps 50 \
       --num-segments 25 \
       --min-seg-len 5 \
       --max-seg-len 25

2) Aggregate samples belonging to the same target class:

   python plot_timing_class_heatmaps.py \
       --input-dir output/timing_12cls_true \
       --group-by target

The primary plot for comparing where the model is sensitive is mean absolute
attribution. Mean signed attribution is also saved to show whether inputs tend
to support (+) or suppress (-) a target class. All class plots share the same
color limits, which is necessary for valid visual comparison.

Color schemes are selectable. Render one, several, or every registered scheme
in a single run:

   python plot_timing_class_heatmaps.py --color-scheme blue
   python plot_timing_class_heatmaps.py --color-scheme blue viridis mono
   python plot_timing_class_heatmaps.py --color-scheme all

A single scheme keeps the flat output layout. Two or more schemes are written
to one subdirectory per scheme. The underlying .npy/.csv data is written once
because it does not depend on the colors.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from dataclasses import replace
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from plot_style import (
    DEFAULT_TIMING_SCHEMES,
    FIGURE_DPI,
    TIMING_SCHEME_NAMES,
    TimingColorScheme,
    apply_paper_style,
    describe_timing_schemes,
    expand_scheme_names,
    resolve_timing_scheme,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("output/timing_12cls_ground_truth"),
        help="Directory produced by explain_timing.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <input-dir>/class_heatmaps_<group-by>",
    )
    parser.add_argument(
        "--group-by",
        choices=("target", "true", "predicted", "correct"),
        default="target",
        help=(
            "target: class whose probability was explained; "
            "true: true-label groups (requires ground-truth target mode); "
            "predicted: predicted-label groups (requires predicted target mode); "
            "correct: correctly classified samples only"
        ),
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="Skip a class when fewer than this many samples are available",
    )
    parser.add_argument(
        "--signed-percentile",
        type=float,
        default=99.0,
        help="Shared symmetric signed-attribution color limit percentile",
    )
    parser.add_argument(
        "--absolute-percentile",
        type=float,
        default=99.0,
        help="Shared absolute-attribution color limit percentile",
    )
    parser.add_argument("--dpi", type=int, default=FIGURE_DPI)
    parser.add_argument(
        "--color-scheme",
        nargs="+",
        default=list(DEFAULT_TIMING_SCHEMES),
        metavar="NAME",
        choices=TIMING_SCHEME_NAMES + ["all", "cvd"],
        help=(
            "One or more color schemes, 'cvd' for every colorblind-safe "
            "scheme, or 'all'. Several schemes are written to one "
            "subdirectory each. The default writes "
            + ", ".join(DEFAULT_TIMING_SCHEMES)
            + " so that protanopia, deuteranopia and tritanopia readers each "
            "have a legible version. Available -- "
            + describe_timing_schemes()
        ),
    )
    parser.add_argument(
        "--signed-cmap",
        default=None,
        help="Override the signed-attribution colormap of every chosen scheme",
    )
    parser.add_argument(
        "--absolute-cmap",
        default=None,
        help="Override the absolute-attribution colormap of every chosen scheme",
    )
    parser.add_argument(
        "--class-time-cmap",
        default=None,
        help="Override the class-by-time overview colormap",
    )
    parser.add_argument(
        "--class-sensor-cmap",
        default=None,
        help="Override the class-by-sensor overview colormap",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        help="Scale every font size relative to the publication defaults",
    )
    parser.add_argument(
        "--sensor-names",
        nargs="*",
        default=None,
        help="Optional names for all channels in packed order: L channels then R channels",
    )
    return parser.parse_args()


def _load_required(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    return np.load(path)


def _safe_percentile(values: Iterable[np.ndarray], percentile: float) -> float:
    flattened = [np.asarray(x, dtype=np.float64).ravel() for x in values]
    flattened = [x[np.isfinite(x)] for x in flattened if x.size > 0]
    flattened = [x for x in flattened if x.size > 0]
    if not flattened:
        return 1e-12
    merged = np.concatenate(flattened)
    limit = float(np.percentile(merged, percentile))
    return max(limit, 1e-12)


def _default_sensor_names(feature_dim: int, sensor_dim: int) -> list[str]:
    channel_names_per_side = ["COP AP", "COP ML", "AP", "ML", "V"]

    if feature_dim == 2 * sensor_dim and sensor_dim == len(channel_names_per_side):
        return channel_names_per_side + channel_names_per_side

    return [f"Feature {index + 1}" for index in range(feature_dim)]


def _plot_joint_heatmap(
    values_td: np.ndarray,
    *,
    class_name: str,
    sample_count: int,
    sensor_names: list[str],
    sensor_dim: int,
    output_path: Path,
    signed: bool,
    color_limit: float,
    dpi: int,
    group_by: str,
    scheme: TimingColorScheme,
) -> None:
    # values_td: [T, D] -> [D, T]
    values_dt = values_td.T

    # 폰트가 커진 만큼 여백도 함께 키웁니다.
    figure_width = max(12.5, values_dt.shape[1] / 8.0)
    figure_height = max(6.0, values_dt.shape[0] * 0.62)
    plt.figure(figsize=(figure_width, figure_height))

    if signed:
        plt.imshow(
            values_dt,
            aspect="auto",
            origin="upper",
            vmin=-color_limit,
            vmax=color_limit,
            cmap=scheme.signed,
        )
        kind = "mean signed"
    else:
        plt.imshow(
            values_dt,
            aspect="auto",
            origin="upper",
            vmin=0.0,
            vmax=color_limit,
            cmap=scheme.absolute,
        )
        kind = "mean absolute"

    plt.xlabel("Time index")
    plt.ylabel("Sensor channel")
    plt.yticks(np.arange(len(sensor_names)), sensor_names)
    plt.title(
        f"{class_name}: {kind} attribution "
        f"(group={group_by}, n={sample_count})"
    )

    if values_dt.shape[0] == 2 * sensor_dim:
        plt.axhline(sensor_dim - 0.5, linewidth=1.6, linestyle="--", color="black")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def _plot_all_classes_time_heatmap(
    class_time: np.ndarray,
    *,
    class_names: list[str],
    counts: np.ndarray,
    output_path: Path,
    vmax: float,
    dpi: int,
    group_by: str,
    cmap: str,
) -> None:
    plt.figure(figsize=(13.0, max(6.5, 0.58 * len(class_names))))
    plt.imshow(
        class_time,
        aspect="auto",
        origin="upper",
        vmin=0.0,
        vmax=vmax,
        cmap=cmap,
    )
    labels = [f"{name} (n={int(count)})" for name, count in zip(class_names, counts)]
    plt.yticks(np.arange(len(class_names)), labels)
    plt.xlabel("Time index")
    plt.ylabel("Class")
    plt.title(f"All classes: mean absolute temporal importance (group={group_by})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def _plot_all_classes_feature_heatmap(
    class_feature: np.ndarray,
    *,
    class_names: list[str],
    counts: np.ndarray,
    sensor_names: list[str],
    output_path: Path,
    vmax: float,
    dpi: int,
    group_by: str,
    cmap: str,
) -> None:
    plt.figure(
        figsize=(
            max(10.0, 1.0 * len(sensor_names)),
            max(6.5, 0.58 * len(class_names)),
        )
    )
    plt.imshow(
        class_feature,
        aspect="auto",
        origin="upper",
        vmin=0.0,
        vmax=vmax,
        cmap=cmap,
    )
    labels = [f"{name} (n={int(count)})" for name, count in zip(class_names, counts)]
    plt.yticks(np.arange(len(class_names)), labels)
    plt.xticks(np.arange(len(sensor_names)), sensor_names, rotation=45, ha="right")
    plt.xlabel("Sensor channel")
    plt.ylabel("Class")
    plt.title(f"All classes: mean absolute sensor importance (group={group_by})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def _apply_cmap_overrides(
    scheme: TimingColorScheme,
    args: argparse.Namespace,
) -> TimingColorScheme:
    """--*-cmap 인자로 scheme의 개별 컬러맵을 덮어씁니다.

    dataclasses.replace를 쓰므로 TimingColorScheme에 필드가 추가되어도
    여기서 빠뜨리지 않습니다. 사용자가 컬러맵을 직접 지정하면 더 이상
    색각이상 검증을 통과한 조합이라고 보장할 수 없으므로 cvd_safe를 내립니다.
    """
    overrides = {
        "signed": args.signed_cmap,
        "absolute": args.absolute_cmap,
        "class_time": args.class_time_cmap,
        "class_sensor": args.class_sensor_cmap,
    }
    applied = {k: v for k, v in overrides.items() if v}
    if not applied:
        return scheme
    return replace(scheme, cvd_safe=False, **applied)


def main() -> None:
    args = parse_args()
    apply_paper_style(scale=args.font_scale)
    if args.min_samples < 1:
        raise ValueError("--min-samples must be at least 1")
    for value, name in (
        (args.signed_percentile, "--signed-percentile"),
        (args.absolute_percentile, "--absolute-percentile"),
    ):
        if not 0.0 < value <= 100.0:
            raise ValueError(f"{name} must be in (0, 100]")

    metadata_path = args.input_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json does not exist in {args.input_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    signed = _load_required(args.input_dir / "timing_signed.npy")
    targets = _load_required(args.input_dir / "targets.npy").astype(np.int64)
    true_labels = _load_required(args.input_dir / "true_labels.npy").astype(np.int64)
    probabilities = _load_required(args.input_dir / "original_probabilities.npy")

    if signed.ndim != 3:
        raise ValueError(f"timing_signed.npy must have shape [N,T,D], got {signed.shape}")
    sample_count, time_steps, feature_dim = signed.shape
    if targets.shape != (sample_count,) or true_labels.shape != (sample_count,):
        raise ValueError("targets.npy and true_labels.npy must have one value per sample")
    if probabilities.ndim != 2 or probabilities.shape[0] != sample_count:
        raise ValueError("original_probabilities.npy must have shape [N,C]")

    class_names = list(metadata["output_classes"])
    num_classes = len(class_names)
    if probabilities.shape[1] != num_classes:
        raise ValueError(
            f"metadata contains {num_classes} classes, but probabilities have "
            f"shape {probabilities.shape}"
        )

    predicted = probabilities.argmax(axis=1).astype(np.int64)
    target_mode = str(metadata.get("target_mode", "unknown"))

    if args.group_by == "target":
        group_labels = targets
    elif args.group_by == "true":
        if target_mode != "ground-truth":
            raise ValueError(
                "--group-by true is only valid when attribution was generated with "
                "--target-mode ground-truth. Otherwise the heatmap would group "
                "attribution for a different target class."
            )
        group_labels = true_labels
    elif args.group_by == "predicted":
        if target_mode != "predicted":
            raise ValueError(
                "--group-by predicted is only valid when attribution was generated "
                "with --target-mode predicted."
            )
        group_labels = predicted
    else:
        # For correct predictions, target=true=predicted in both target modes.
        group_labels = true_labels

    sensor_dim = int(metadata.get("sensor_dim_per_side", feature_dim // 2))
    if args.sensor_names is None:
        sensor_names = _default_sensor_names(feature_dim, sensor_dim)
    else:
        sensor_names = list(args.sensor_names)
        if len(sensor_names) != feature_dim:
            raise ValueError(
                f"--sensor-names requires exactly {feature_dim} names, "
                f"but {len(sensor_names)} were supplied"
            )

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.input_dir / f"class_heatmaps_{args.group_by}"
    output_dir.mkdir(parents=True, exist_ok=True)

    scheme_names = expand_scheme_names(args.color_scheme)
    # Scheme 하나면 기존 경로 그대로, 둘 이상이면 scheme별 하위 디렉토리를 씁니다.
    use_scheme_subdirs = len(scheme_names) > 1

    mean_signed = np.full(
        (num_classes, time_steps, feature_dim),
        np.nan,
        dtype=np.float32,
    )
    mean_absolute = np.full_like(mean_signed, np.nan)
    std_absolute = np.full_like(mean_signed, np.nan)
    counts = np.zeros(num_classes, dtype=np.int64)

    for class_index in range(num_classes):
        mask = group_labels == class_index
        if args.group_by == "correct":
            mask &= predicted == true_labels

        class_count = int(mask.sum())
        counts[class_index] = class_count
        if class_count < args.min_samples:
            continue

        class_attr = signed[mask]
        mean_signed[class_index] = class_attr.mean(axis=0)
        class_abs = np.abs(class_attr)
        mean_absolute[class_index] = class_abs.mean(axis=0)
        std_absolute[class_index] = class_abs.std(axis=0)

    valid_class_indices = [
        index for index, count in enumerate(counts) if count >= args.min_samples
    ]
    if not valid_class_indices:
        raise RuntimeError(
            "No class has enough samples. Use a lower --min-samples value or "
            "generate attribution for more samples."
        )

    signed_limit = _safe_percentile(
        [np.abs(mean_signed[index]) for index in valid_class_indices],
        args.signed_percentile,
    )
    absolute_limit = _safe_percentile(
        [mean_absolute[index] for index in valid_class_indices],
        args.absolute_percentile,
    )

    # Class x time overview. Missing classes remain NaN and appear blank.
    class_time_abs = np.nanmean(mean_absolute, axis=-1)
    class_feature_abs = np.nanmean(mean_absolute, axis=1)
    time_limit = _safe_percentile(
        [class_time_abs[index] for index in valid_class_indices],
        args.absolute_percentile,
    )
    feature_limit = _safe_percentile(
        [class_feature_abs[index] for index in valid_class_indices],
        args.absolute_percentile,
    )

    # 색상만 다르고 데이터는 같으므로, 계산은 한 번 하고 렌더링만 반복합니다.
    scheme_dirs: dict[str, str] = {}
    for scheme_name in scheme_names:
        scheme = _apply_cmap_overrides(resolve_timing_scheme(scheme_name), args)
        scheme_dir = output_dir / scheme_name if use_scheme_subdirs else output_dir
        signed_dir = scheme_dir / "mean_signed"
        absolute_dir = scheme_dir / "mean_absolute"
        signed_dir.mkdir(parents=True, exist_ok=True)
        absolute_dir.mkdir(parents=True, exist_ok=True)
        scheme_dirs[scheme_name] = str(scheme_dir)

        for class_index in valid_class_indices:
            class_name = class_names[class_index]
            safe_name = class_name.replace("/", "_").replace(" ", "_")
            _plot_joint_heatmap(
                mean_signed[class_index],
                class_name=class_name,
                sample_count=int(counts[class_index]),
                sensor_names=sensor_names,
                sensor_dim=sensor_dim,
                output_path=signed_dir / f"{class_index:02d}_{safe_name}_mean_signed.png",
                signed=True,
                color_limit=signed_limit,
                dpi=args.dpi,
                group_by=args.group_by,
                scheme=scheme,
            )
            _plot_joint_heatmap(
                mean_absolute[class_index],
                class_name=class_name,
                sample_count=int(counts[class_index]),
                sensor_names=sensor_names,
                sensor_dim=sensor_dim,
                output_path=absolute_dir / f"{class_index:02d}_{safe_name}_mean_absolute.png",
                signed=False,
                color_limit=absolute_limit,
                dpi=args.dpi,
                group_by=args.group_by,
                scheme=scheme,
            )

        _plot_all_classes_time_heatmap(
            class_time_abs,
            class_names=class_names,
            counts=counts,
            output_path=scheme_dir / "all_classes_time_importance_mean_absolute.png",
            vmax=time_limit,
            dpi=args.dpi,
            group_by=args.group_by,
            cmap=scheme.class_time,
        )
        _plot_all_classes_feature_heatmap(
            class_feature_abs,
            class_names=class_names,
            counts=counts,
            sensor_names=sensor_names,
            output_path=scheme_dir / "all_classes_sensor_importance_mean_absolute.png",
            vmax=feature_limit,
            dpi=args.dpi,
            group_by=args.group_by,
            cmap=scheme.class_sensor,
        )

    np.save(output_dir / "class_mean_signed.npy", mean_signed)
    np.save(output_dir / "class_mean_absolute.npy", mean_absolute)
    np.save(output_dir / "class_std_absolute.npy", std_absolute)
    np.save(output_dir / "class_counts.npy", counts)
    np.save(output_dir / "class_time_importance_mean_absolute.npy", class_time_abs)
    np.save(output_dir / "class_sensor_importance_mean_absolute.npy", class_feature_abs)

    with (output_dir / "class_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "class_index",
                "class_name",
                "sample_count",
                "heatmap_created",
                "group_by",
                "attribution_target_mode",
            ]
        )
        for class_index, class_name in enumerate(class_names):
            writer.writerow(
                [
                    class_index,
                    class_name,
                    int(counts[class_index]),
                    bool(counts[class_index] >= args.min_samples),
                    args.group_by,
                    target_mode,
                ]
            )

    summary = {
        "input_dir": str(args.input_dir),
        "output_dir": str(output_dir),
        "group_by": args.group_by,
        "attribution_target_mode": target_mode,
        "num_samples": int(sample_count),
        "input_shape": [int(x) for x in signed.shape],
        "class_names": class_names,
        "class_counts": {
            name: int(count) for name, count in zip(class_names, counts)
        },
        "min_samples": int(args.min_samples),
        # 컬러바를 그리지 않으므로 색 범위를 여기에 남깁니다.
        "shared_signed_color_limit": signed_limit,
        "shared_absolute_color_limit": absolute_limit,
        "all_classes_time_color_limit": float(time_limit),
        "all_classes_sensor_color_limit": float(feature_limit),
        "color_limit_note": (
            "Figures carry no colorbar. Signed maps span "
            "[-shared_signed_color_limit, +shared_signed_color_limit]; "
            "absolute maps span [0, the matching limit]."
        ),
        "color_schemes": scheme_names,
        "color_scheme_dirs": scheme_dirs,
        "dpi": int(args.dpi),
        "font_scale": float(args.font_scale),
        "primary_recommended_output": str(
            Path(scheme_dirs[scheme_names[0]])
            / "all_classes_time_importance_mean_absolute.png"
        ),
    }
    (output_dir / "class_heatmap_metadata.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
