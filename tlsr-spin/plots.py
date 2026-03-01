"""
Visualization for TLSR spin-orbit dynamics.

Reimplements DayNite's plotSpinReg.py, plotRegHistFITfunc.py, and
plotRegPie.py with shared/plotting.py style.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch

from shared.constants import YEAR
from shared.plotting import apply_style, save_and_show
from .regime_classifier import RegimeResult, RegimeType


# Regime color scheme
REGIME_COLORS: dict[str, str] = {
    RegimeType.TL_ZERO.value: "#2196F3",   # blue — substellar lock
    RegimeType.TL_PI.value: "#FF9800",     # orange — antistellar lock
    RegimeType.SPINNING.value: "#4CAF50",  # green — spinning
    RegimeType.PTB.value: "#9E9E9E",       # grey — transitional
}

REGIME_LABELS: dict[str, str] = {
    RegimeType.TL_ZERO.value: "TL Zero (substellar)",
    RegimeType.TL_PI.value: "TL Pi (antistellar)",
    RegimeType.SPINNING.value: "Spinning",
    RegimeType.PTB.value: "PTB (transitional)",
}


def plot_spin_history(
    t: np.ndarray,
    gamma: np.ndarray,
    result: RegimeResult,
    output_path: str,
    title: str = "Spin-Orbit History",
) -> None:
    """Plot γ(t) with regime classification overlay.

    Reproduces Shakespeare & Steffen 2023 Fig B2 style.

    Args:
        t: Time array (seconds).
        gamma: Spin-orbit angle γ (radians).
        result: RegimeResult with classified regimes.
        output_path: Path to save figure.
        title: Plot title.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(14, 5))

    t_yr = t / YEAR

    # Background regime shading
    for regime in result.regimes:
        color = REGIME_COLORS.get(regime.type.value, "#CCCCCC")
        ax.axvspan(
            regime.t_start / YEAR, regime.t_end / YEAR,
            alpha=0.15, color=color, linewidth=0,
        )

    # γ(t) trace
    ax.plot(t_yr, gamma, color="black", linewidth=0.3, rasterized=True)

    # Reference lines
    ax.axhline(0, color="#2196F3", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.axhline(np.pi, color="#FF9800", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.axhline(-np.pi, color="#FF9800", linestyle="--", alpha=0.4, linewidth=0.8)

    ax.set_xlabel("Time (yr)")
    ax.set_ylabel("γ (rad)")
    ax.set_title(title)

    # Legend for regime colors
    legend_patches = [
        Patch(facecolor=color, alpha=0.3, label=REGIME_LABELS[name])
        for name, color in REGIME_COLORS.items()
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8)

    save_and_show(fig, output_path)


def plot_regime_histogram(
    regime_stats: dict[str, dict[str, float]],
    output_path: str,
    fit_power_law: bool = True,
    title: str = "Regime Duration Distribution",
) -> None:
    """Plot histogram of regime durations by type.

    Reproduces Shakespeare & Steffen 2023 Fig 3 style.

    Args:
        regime_stats: From compute_regime_stats().
        output_path: Path to save figure.
        fit_power_law: Whether to overlay a power-law fit.
        title: Plot title.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar chart of mean durations with error bars
    types_with_data = [
        (name, stats) for name, stats in regime_stats.items()
        if stats["count"] > 0
    ]
    if not types_with_data:
        ax.text(0.5, 0.5, "No regime data", ha="center", va="center",
                transform=ax.transAxes)
        save_and_show(fig, output_path)
        return

    x = np.arange(len(types_with_data))
    labels = [REGIME_LABELS.get(name, name) for name, _ in types_with_data]
    means = [s["mean_yr"] for _, s in types_with_data]
    stds = [s["std_yr"] for _, s in types_with_data]
    colors = [REGIME_COLORS.get(name, "#999999") for name, _ in types_with_data]

    ax.bar(x, means, yerr=stds, color=colors, alpha=0.8, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Mean Duration (yr)")
    ax.set_title(title)

    # Annotate counts
    for i, (name, stats) in enumerate(types_with_data):
        ax.annotate(
            f"n={stats['count']:.0f}",
            (i, means[i] + stds[i]),
            ha="center", va="bottom", fontsize=9,
        )

    save_and_show(fig, output_path)


def plot_regime_pie(
    regime_fractions: dict[str, float],
    output_path: str,
    title: str = "Time in Each Regime",
) -> None:
    """Pie chart of time fraction in each regime.

    Reproduces Shakespeare & Steffen 2023 Fig 4 style.

    Args:
        regime_fractions: From compute_regime_fractions().
        output_path: Path to save figure.
        title: Plot title.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 8))

    # Filter out zero-fraction regimes
    data = {
        name: frac for name, frac in regime_fractions.items()
        if frac > 0.001
    }
    if not data:
        ax.text(0.5, 0.5, "No regime data", ha="center", va="center",
                transform=ax.transAxes)
        save_and_show(fig, output_path)
        return

    labels = [REGIME_LABELS.get(name, name) for name in data]
    sizes = list(data.values())
    colors = [REGIME_COLORS.get(name, "#999999") for name in data]

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, textprops={"fontsize": 11},
    )
    for autotext in autotexts:
        autotext.set_fontsize(10)

    ax.set_title(title)
    save_and_show(fig, output_path)


def plot_regime_summary(
    results: dict[str, dict],
    output_path: str,
    title: str = "TLSR Regime Comparison",
) -> None:
    """Combined overview comparing multiple simulation runs.

    Args:
        results: Dict mapping run label to dict with 'fractions' and 'stats' keys.
        output_path: Path to save figure.
        title: Plot title.
    """
    apply_style()
    n_runs = len(results)
    if n_runs == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: stacked bar of regime fractions
    ax = axes[0]
    run_labels = list(results.keys())
    x = np.arange(n_runs)
    bottoms = np.zeros(n_runs)

    for rtype in RegimeType:
        fracs = [
            results[label].get("fractions", {}).get(rtype.value, 0.0)
            for label in run_labels
        ]
        color = REGIME_COLORS[rtype.value]
        ax.bar(x, fracs, bottom=bottoms, color=color, alpha=0.8,
               label=REGIME_LABELS[rtype.value])
        bottoms += np.array(fracs)

    ax.set_xticks(x)
    ax.set_xticklabels(run_labels, rotation=30, ha="right")
    ax.set_ylabel("Time Fraction")
    ax.set_title("Regime Fractions")
    ax.legend(fontsize=8)

    # Right: quasi-stable fraction
    ax = axes[1]
    qs_fracs = [
        results[label].get("quasi_stable_fraction", 0.0)
        for label in run_labels
    ]
    ax.bar(x, qs_fracs, color="#607D8B", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(run_labels, rotation=30, ha="right")
    ax.set_ylabel("Quasi-Stable Fraction")
    ax.set_title("Long-Duration Regime Fraction (>900 yr)")

    fig.suptitle(title)
    fig.tight_layout()
    save_and_show(fig, output_path)


def plot_phase_portrait(
    t: np.ndarray,
    gamma: np.ndarray,
    gamma_dot: np.ndarray,
    intervals: list[dict],
    output_path: str,
    max_panels: int = 6,
) -> None:
    """Phase portrait (γ̇ vs γ) for individual PTB episodes.

    Args:
        t: Time array (seconds).
        gamma: Spin-orbit angle γ (radians).
        gamma_dot: γ̇ (rad/s).
        intervals: From extract_ptb_intervals().
        output_path: Path to save figure.
        max_panels: Maximum number of panels to plot.
    """
    if not intervals:
        return

    # Select longest PTB episodes
    sorted_ivs = sorted(intervals, key=lambda x: x["duration_yr"], reverse=True)
    selected = sorted_ivs[:max_panels]

    apply_style()
    ncols = min(3, len(selected))
    nrows = (len(selected) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)

    for idx, interval in enumerate(selected):
        ax = axes[idx // ncols][idx % ncols]

        # Find time indices for this interval
        mask = (t >= interval["t_start"]) & (t <= interval["t_end"])
        if mask.sum() < 2:
            ax.set_visible(False)
            continue

        g = gamma[mask]
        gd = gamma_dot[mask]
        t_seg = t[mask]

        # Wrap γ to [-π, π] for display
        g_wrapped = (g + np.pi) % (2 * np.pi) - np.pi

        # Time-colormapped trajectory
        t_norm = (t_seg - t_seg[0]) / max(t_seg[-1] - t_seg[0], 1.0)
        points = np.column_stack([g_wrapped, gd]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap="viridis", linewidth=0.8)
        lc.set_array(t_norm[:-1])
        ax.add_collection(lc)

        ax.autoscale()

        # Mark attractors
        ax.plot(0, 0, "o", color="#2196F3", markersize=8, zorder=5, label="TL Zero")
        ax.plot(np.pi, 0, "o", color="#FF9800", markersize=8, zorder=5, label="TL Pi")
        ax.plot(-np.pi, 0, "o", color="#FF9800", markersize=8, zorder=5)

        ax.set_xlabel("γ (rad)")
        ax.set_ylabel("γ̇ (rad/s)")
        dur = interval["duration_yr"]
        utype = interval["underlying_type"]
        ax.set_title(f"PTB #{interval['index']} ({dur:.1f} yr, was {utype})", fontsize=10)

        if idx == 0:
            ax.legend(fontsize=7, loc="upper right")

    # Hide unused panels
    for idx in range(len(selected), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("PTB Phase Portraits (γ̇ vs γ)")
    fig.tight_layout()
    save_and_show(fig, output_path)


def plot_ptb_zoom(
    t: np.ndarray,
    gamma: np.ndarray,
    result: RegimeResult,
    intervals: list[dict],
    output_path: str,
    max_panels: int = 4,
    margin_yr: float = 50.0,
) -> None:
    """Zoomed time-series of γ(t) around individual PTB episodes.

    Args:
        t: Time array (seconds).
        gamma: Spin-orbit angle γ (radians).
        result: RegimeResult with classified regimes.
        intervals: From extract_ptb_intervals().
        output_path: Path to save figure.
        max_panels: Maximum number of panels.
        margin_yr: Context margin before/after PTB (years).
    """
    if not intervals:
        return

    sorted_ivs = sorted(intervals, key=lambda x: x["duration_yr"], reverse=True)
    selected = sorted_ivs[:max_panels]

    apply_style()
    fig, axes = plt.subplots(len(selected), 1, figsize=(14, 4 * len(selected)), squeeze=False)

    margin_s = margin_yr * YEAR

    for idx, interval in enumerate(selected):
        ax = axes[idx][0]

        t_lo = interval["t_start"] - margin_s
        t_hi = interval["t_end"] + margin_s
        mask = (t >= t_lo) & (t <= t_hi)

        if mask.sum() < 2:
            ax.set_visible(False)
            continue

        t_seg = t[mask] / YEAR
        g_seg = gamma[mask]

        # Regime shading
        for regime in result.regimes:
            if regime.t_end < t_lo or regime.t_start > t_hi:
                continue
            color = REGIME_COLORS.get(regime.type.value, "#CCCCCC")
            ax.axvspan(
                max(regime.t_start, t_lo) / YEAR,
                min(regime.t_end, t_hi) / YEAR,
                alpha=0.2, color=color, linewidth=0,
            )

        ax.plot(t_seg, g_seg, color="black", linewidth=0.5, rasterized=True)

        # Reference lines
        ax.axhline(0, color="#2196F3", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.axhline(np.pi, color="#FF9800", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.axhline(-np.pi, color="#FF9800", linestyle="--", alpha=0.4, linewidth=0.8)

        ax.set_xlabel("Time (yr)")
        ax.set_ylabel("γ (rad)")
        dur = interval["duration_yr"]
        utype = interval["underlying_type"]
        ax.set_title(f"PTB #{interval['index']} ({dur:.1f} yr, was {utype})", fontsize=10)

    # Shared legend on first panel
    legend_patches = [
        Patch(facecolor=color, alpha=0.3, label=REGIME_LABELS[name])
        for name, color in REGIME_COLORS.items()
    ]
    axes[0][0].legend(handles=legend_patches, loc="upper right", fontsize=8)

    fig.suptitle("PTB Zoom-In: γ(t) Time Series")
    fig.tight_layout()
    save_and_show(fig, output_path)
