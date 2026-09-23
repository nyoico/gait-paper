"""Publication-quality matplotlib styling shared by every plotting script.

이 모듈은 저장소 전체의 그림 스타일을 한 곳에서 관리합니다.

- 모든 그림은 300 dpi로 저장됩니다.
- 폰트 크기는 title=18, label=16, tick=14로 고정되어 있습니다.
- 데이터 곡선의 선 굵기는 3입니다.
- 컬러바는 사용하지 않습니다. 값의 범위는 각 스크립트가 함께 저장하는
  JSON/CSV의 color limit 항목에서 확인합니다.
- Confusion matrix는 블루 계열 컬러맵을 사용합니다.
- TIMING attribution 히트맵은 색각이상(CVD)에서도 읽히도록 검증된
  여러 color scheme으로 동시에 출력합니다.

사용법::

    from plot_style import apply_paper_style
    apply_paper_style()

TIMING 히트맵 색상은 ``resolve_timing_scheme("cividis")`` 처럼 이름으로 가져옵니다.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib as mpl


# 저장 해상도입니다. 저장소 내 모든 savefig는 이 값을 따릅니다.
FIGURE_DPI = 300

# Confusion matrix 전용 블루 계열 컬러맵입니다.
CONFUSION_CMAP = "Blues"


# ---------------------------------------------------------------------------
# 폰트 크기
# ---------------------------------------------------------------------------
# 저장소 전체에서 이 세 값만 쓰도록 통일했습니다. 개별 스크립트에
# 숫자를 직접 적지 말고 여기에서 가져다 쓰십시오.
TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 14

# 데이터 곡선의 선 굵기입니다. 축 테두리/격자/눈금 굵기와는 다릅니다.
LINE_WIDTH = 3

# Confusion matrix 셀 안에 찍는 숫자입니다. 셀이 넓어서 눈금 크기로는
# 작아 보이므로 축 이름과 같은 크기로 둡니다.
CONFUSION_CELL_FONTSIZE = LABEL_FONTSIZE


PAPER_RCPARAMS: dict[str, object] = {
    # 해상도
    "figure.dpi": 150,          # 화면 표시용. 저장 해상도는 savefig.dpi입니다.
    "savefig.dpi": FIGURE_DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,

    # 폰트
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

    # 축과 선
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

    # 범례
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.borderpad": 0.5,

    # 눈금 방향
    "xtick.direction": "out",
    "ytick.direction": "out",

    # 벡터 출력 시 TrueType 폰트를 임베드합니다(대부분의 학회 요구사항).
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_paper_style(scale: float = 1.0, **overrides: object) -> None:
    """논문용 rcParams를 전역 적용합니다.

    scale을 주면 모든 폰트 크기가 비례해서 커지거나 작아집니다.
    overrides로 개별 rcParams를 덮어쓸 수 있습니다.
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
    """폰트 상수에 --font-scale을 적용합니다."""
    return round(float(size) * float(scale), 1)


def confusion_text_color(value: float, vmax: float, threshold: float = 0.55) -> str:
    """블루 계열 셀 위에서 숫자가 항상 읽히도록 글자색을 고릅니다.

    값이 큰(=진한) 셀에는 흰색, 나머지에는 검은색을 씁니다.
    """
    if vmax <= 0:
        return "black"
    return "white" if float(value) / float(vmax) >= threshold else "black"


# ---------------------------------------------------------------------------
# TIMING 히트맵 color scheme
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TimingColorScheme:
    """TIMING attribution 그림 한 세트에 쓰이는 컬러맵 모음입니다."""

    name: str
    description: str
    signed: str        # 부호 있는 attribution (0을 중심으로 발산형)
    absolute: str      # 절댓값 attribution (순차형)
    class_time: str    # 클래스 x 시간 요약 히트맵
    class_sensor: str  # 클래스 x 센서 요약 히트맵
    cvd_safe: bool     # 색각이상 시뮬레이션 검증을 통과했는가


# 아래 조합은 Machado et al.(2009) severity 1.0 행렬로 protanopia,
# deuteranopia, tritanopia를 시뮬레이션한 뒤 CIE Lab 거리로 검증했습니다.
#   - 순차형(absolute): 11단계 인접 색 최소 dE와 밝기 단조성
#   - 발산형(signed)  : 중심 대칭 쌍(-t, +t)의 최소 dE = 부호 구분력
# 세 가지 색각이상 유형에서 정상시야 대비 45% 이상을 유지하는 것만 남겼습니다.
# 검증에서 탈락해 제외한 조합: Spectral_r(유지율 24%), RdGy_r(32%),
# plasma(41%), YlGnBu(44%).
TIMING_COLOR_SCHEMES: dict[str, TimingColorScheme] = {
    "default": TimingColorScheme(
        name="default",
        description=(
            "RdBu_r + viridis. 가장 무난한 기본값이며 세 유형 모두에서 "
            "부호 구분력을 76% 유지한다"
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
            "PuOr_r + cividis. 적록색약(protan/deutan)에 가장 강하다"
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
            "managua + YlOrRd. 세 유형 평균 유지율이 가장 높고 "
            "청황색약(tritan)에 특히 강하다"
        ),
        signed="managua",
        absolute="YlOrRd",
        class_time="YlOrRd",
        class_sensor="YlOrRd",
        cvd_safe=True,
    ),
    "berlin": TimingColorScheme(
        name="berlin",
        description="berlin + magma. Crameri 계열로 지각적으로 균일하다",
        signed="berlin",
        absolute="magma",
        class_time="magma",
        class_sensor="magma",
        cvd_safe=True,
    ),
    "vanimo": TimingColorScheme(
        name="vanimo",
        description="vanimo + inferno. Crameri 계열, berlin과 다른 색상축",
        signed="vanimo",
        absolute="inferno",
        class_time="inferno",
        class_sensor="inferno",
        cvd_safe=True,
    ),
    "blue": TimingColorScheme(
        name="blue",
        description=(
            "RdBu_r + Blues. 단일 색상 계열이라 흑백 인쇄에서도 "
            "절댓값 그림이 그대로 읽힌다"
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
            "RdGy_r + Greys. 흑백 인쇄 전용. 절댓값 그림에만 쓰십시오. "
            "발산형은 밝기가 중심 대칭이라 흑백에서 부호를 구분할 수 없고 "
            "CVD 검증도 통과하지 못한다"
        ),
        signed="RdGy_r",
        absolute="Greys",
        class_time="Greys",
        class_sensor="Greys",
        cvd_safe=False,
    ),
}

# --color-scheme 에서 고를 수 있는 전체 이름 목록입니다.
TIMING_SCHEME_NAMES: list[str] = list(TIMING_COLOR_SCHEMES)

# CVD 검증을 통과한 scheme 이름입니다. "cvd" 별칭이 이 목록으로 펼쳐집니다.
CVD_SAFE_SCHEME_NAMES: list[str] = [
    name for name, scheme in TIMING_COLOR_SCHEMES.items() if scheme.cvd_safe
]

# 인자를 주지 않았을 때 생성하는 기본 scheme들입니다.
# 세 가지 색각이상 유형을 서로 보완하도록 골랐습니다.
#   default  - 세 유형에서 고르게 안정적
#   cividis  - protan/deutan에 가장 강함
#   managua  - tritan에 가장 강함
DEFAULT_TIMING_SCHEMES: list[str] = ["default", "cividis", "managua"]


def resolve_timing_scheme(name: str) -> TimingColorScheme:
    """이름으로 TIMING color scheme을 가져옵니다."""
    try:
        return TIMING_COLOR_SCHEMES[name]
    except KeyError:
        raise ValueError(
            f"Unknown TIMING color scheme: {name!r}. "
            f"Available: {', '.join(TIMING_SCHEME_NAMES)}"
        ) from None


def expand_scheme_names(names: list[str] | None) -> list[str]:
    """--color-scheme 인자를 실제 scheme 이름 목록으로 펼칩니다.

    - 비어 있으면 DEFAULT_TIMING_SCHEMES
    - "all"이 있으면 등록된 전체
    - "cvd"가 있으면 CVD 검증을 통과한 것 전체
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
    """scheme 설명 문자열을 만듭니다.

    argparse는 help 문자열에 %-포맷을 적용하므로 설명에 들어 있는
    퍼센트 기호를 기본적으로 %%로 이스케이프합니다.
    """
    text = "; ".join(
        f"{scheme.name}: {scheme.description}"
        for scheme in TIMING_COLOR_SCHEMES.values()
    )
    return text.replace("%", "%%") if for_argparse else text
