"""Publication-quality matplotlib styling shared by every plotting script.

This module centralizes figure styling across the repository.

- All figures are saved at 300 dpi.
- Font sizes are fixed at title=18, label=16, and tick=14.
- Data curves use a line width of 3.
- Colorbars are omitted. Value ranges are recorded in the color-limit fields
  of the JSON/CSV files saved by the corresponding scripts.
- Confusion matrices use a blue colormap.
- TIMING attribution heatmaps are rendered in multiple color schemes
  validated for readability under color vision deficiency (CVD).

Usage::

    from plot_style import apply_paper_style
    apply_paper_style()

Retrieve TIMING heatmap colors by name, e.g., ``resolve_timing_scheme("cividis")``.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib as mpl


# Output resolution used by savefig throughout the repository.
FIGURE_DPI = 300

# Blue colormap reserved for confusion matrices.
CONFUSION_CMAP = "Blues"


# ---------------------------------------------------------------------------
# Font sizes
# ---------------------------------------------------------------------------
# Use these three values consistently throughout the repository.
# Import them here instead of hardcoding sizes in individual scripts.
TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 14

# Line width for data curves, separate from axes, grid, and tick widths.
LINE_WIDTH = 3

# Annotation size for confusion matrix cells. Use the axis-label size because
# the tick-label size appears too small in these relatively large cells.
CONFUSION_CELL_FONTSIZE = LABEL_FONTSIZE


PAPER_RCPARAMS: dict[str, object] = {
    # Resolution
    "figure.dpi": 150,          # Display resolution; savefig.dpi controls output.
    "savefig.dpi": FIGURE_DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,

    # Fonts
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": TICK_FONTSIZE,
    "axes.titlesize": TITLE_FONTSIZE,
    "axes.labelsize": LABEL_FONTSIZE,
    "xtick.labelsize": TICK_FONTSIZE,
    "ytick.labelsize": TICK_FONTSIZE,
    "legend.fontsize": TICK_FONTSIZE,
    "legend.title_fontsize": LABEL_FONTSIZE,
    "figure.titlesize": TITLE_FONTSIZE,

    # Axes and lines
    "axes.titleweight": "bold",
    "axes.labelweight": "normal",
    "axes.linewidth": 1.3,
    "lines.linewidth": LINE_WIDTH,
    "lines.markersize": 7,
    "xtick.major.width": 1.3,
    "ytick.major.width": 1.3,
    "xtick.major.size": 5.5,
    "ytick.major.size": 5.5,
    "grid.linewidth": 0.8,

    # Legend
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.borderpad": 0.5,

    # Tick direction
    "xtick.direction": "out",
    "ytick.direction": "out",

    # Embed TrueType fonts in vector output, as required by many conferences.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_paper_style(scale: float = 1.0, **overrides: object) -> None:
    """Apply publication-oriented rcParams globally.

    scale adjusts all font sizes proportionally.
    overrides can replace individual rcParams values.
    """
    params = dict(PAPER_RCPARAMS)

    if scale != 1.0:
        font_keys = (
            "font.size",
            "axes.titlesize",
            "axes.labelsize",
            "xtick.labelsize",
            "ytick.labelsize",
            "legend.fontsize",
            "legend.title_fontsize",
            "figure.titlesize",
        )
        for key in font_keys:
            params[key] = round(float(params[key]) * scale, 1)

    params.update(overrides)
    mpl.rcParams.update(params)


def scaled(size: float, scale: float = 1.0) -> float:
    """Apply --font-scale to a font-size constant."""
    return round(float(size) * float(scale), 1)


def confusion_text_color(value: float, vmax: float, threshold: float = 0.55) -> str:
    """Choose a readable annotation color for blue confusion matrix cells.

    Use white for high-value (dark) cells and black for the remaining cells.
    """
    if vmax <= 0:
        return "black"
    return "white" if float(value) / float(vmax) >= threshold else "black"


# ---------------------------------------------------------------------------
# TIMING heatmap color schemes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TimingColorScheme:
    """Define the colormaps used for one set of TIMING attribution figures."""

    name: str
    description: str
    signed: str        # Signed attribution (diverging around zero)
    absolute: str      # Absolute attribution (sequential)
    class_time: str    # Class-by-time summary heatmap
    class_sensor: str  # Class-by-sensor summary heatmap
    cvd_safe: bool     # Whether the scheme passed CVD simulation validation


# These combinations were validated using CIE Lab distances after simulating
# protanopia, deuteranopia, and tritanopia with the severity 1.0 matrices of
# Machado et al. (2009).
#   - Sequential (absolute): minimum adjacent-color dE over 11 levels and luminance monotonicity
#   - Diverging (signed): minimum dE of symmetric pairs (-t, +t), indicating sign discriminability
# Retained schemes preserve at least 45% of normal-vision discriminability across all three CVD types.
# Excluded combinations: Spectral_r (24% retained), RdGy_r (32%),
# plasma (41%), YlGnBu (44%).
TIMING_COLOR_SCHEMES: dict[str, TimingColorScheme] = {
    "default": TimingColorScheme(
        name="default",
        description=(
            "RdBu_r + viridis. A general-purpose default that preserves "
            "76% of sign discriminability across all three CVD types"
        ),
        signed="RdBu_r",
        absolute="viridis",
        class_time="viridis",
        class_sensor="viridis",
        cvd_safe=True,
    ),
    "cividis": TimingColorScheme(
        name="cividis",
        description=(
            "PuOr_r + cividis. Most robust for red-green CVD (protan/deutan)"
        ),
        signed="PuOr_r",
        absolute="cividis",
        class_time="cividis",
        class_sensor="cividis",
        cvd_safe=True,
    ),
    "managua": TimingColorScheme(
        name="managua",
        description=(
            "managua + YlOrRd. Highest mean retention across the three CVD types, "
            "with particular robustness for blue-yellow CVD (tritan)"
        ),
        signed="managua",
        absolute="YlOrRd",
        class_time="YlOrRd",
        class_sensor="YlOrRd",
        cvd_safe=True,
    ),
    "berlin": TimingColorScheme(
        name="berlin",
        description="berlin + magma. Perceptually uniform Crameri colormap family",
        signed="berlin",
        absolute="magma",
        class_time="magma",
        class_sensor="magma",
        cvd_safe=True,
    ),
    "vanimo": TimingColorScheme(
        name="vanimo",
        description="vanimo + inferno. Crameri colormap family with a different hue axis from berlin",
        signed="vanimo",
        absolute="inferno",
        class_time="inferno",
        class_sensor="inferno",
        cvd_safe=True,
    ),
    "blue": TimingColorScheme(
        name="blue",
        description=(
            "RdBu_r + Blues. The single-hue absolute-attribution colormap "
            "remains readable in grayscale printing"
        ),
        signed="RdBu_r",
        absolute="Blues",
        class_time="PuBu",
        class_sensor="Blues",
        cvd_safe=True,
    ),
    "mono": TimingColorScheme(
        name="mono",
        description=(
            "RdGy_r + Greys. Intended for grayscale printing; use only for "
            "absolute-attribution figures. The diverging colormap has symmetric "
            "luminance, so signs cannot be distinguished in grayscale; it also "
            "fails CVD validation"
        ),
        signed="RdGy_r",
        absolute="Greys",
        class_time="Greys",
        class_sensor="Greys",
        cvd_safe=False,
    ),
}

# All names available for --color-scheme.
TIMING_SCHEME_NAMES: list[str] = list(TIMING_COLOR_SCHEMES)

# Schemes that passed CVD validation. The "cvd" alias expands to this list.
CVD_SAFE_SCHEME_NAMES: list[str] = [
    name for name, scheme in TIMING_COLOR_SCHEMES.items() if scheme.cvd_safe
]

# Default schemes generated when no arguments are supplied.
# Selected to provide complementary readability across the three CVD types.
#   default  - consistently robust across all three types
#   cividis  - most robust for protan/deutan
#   managua  - most robust for tritan
DEFAULT_TIMING_SCHEMES: list[str] = ["default", "cividis", "managua"]


def resolve_timing_scheme(name: str) -> TimingColorScheme:
    """Retrieve a TIMING color scheme by name."""
    try:
        return TIMING_COLOR_SCHEMES[name]
    except KeyError:
        raise ValueError(
            f"Unknown TIMING color scheme: {name!r}. "
            f"Available: {', '.join(TIMING_SCHEME_NAMES)}"
        ) from None


def expand_scheme_names(names: list[str] | None) -> list[str]:
    """Expand --color-scheme arguments into a list of scheme names.

    - Empty input selects DEFAULT_TIMING_SCHEMES.
    - "all" selects every registered scheme.
    - "cvd" selects every scheme that passed CVD validation.
    """
    if not names:
        return list(DEFAULT_TIMING_SCHEMES)

    expanded: list[str] = []
    for name in names:
        if name == "all":
            expanded.extend(TIMING_SCHEME_NAMES)
        elif name == "cvd":
            expanded.extend(CVD_SAFE_SCHEME_NAMES)
        else:
            resolve_timing_scheme(name)
            expanded.append(name)

    seen: list[str] = []
    for name in expanded:
        if name not in seen:
            seen.append(name)
    return seen


def describe_timing_schemes(for_argparse: bool = True) -> str:
    """Build a string describing the color schemes.

    argparse applies %-formatting to help strings, so percentage signs
    in the descriptions are escaped as %% by default.
    """
    text = "; ".join(
        f"{scheme.name}: {scheme.description}"
        for scheme in TIMING_COLOR_SCHEMES.values()
    )
    return text.replace("%", "%%") if for_argparse else text
