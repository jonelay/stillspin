"""
Regime classification for spin-orbit dynamics.

Reimplements DayNite's compressRegime.py + calcRegimes.py as a clean
pipeline: extract extrema of γ(t), identify contiguous regimes
(SPINNING, TL_ZERO, TL_PI, PTB), and compute statistics.

Regime definitions (Shakespeare & Steffen 2023):
- TL_ZERO: |γ| < threshold, librating around γ = 0 (substellar lock)
- TL_PI: |γ - π| < threshold (OR |γ + π| < threshold), librating around π
- SPINNING: γ monotonically increasing or decreasing (prograde/retrograde)
- PTB (Perturbed/Transitional Behavior): regime duration < min_regime_yr

The libration threshold is 2 radians (following the paper), which
captures the full libration island width.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from shared.constants import YEAR


class RegimeType(Enum):
    """Spin-orbit regime classification."""
    TL_ZERO = "TL_ZERO"
    TL_PI = "TL_PI"
    SPINNING = "SPINNING"
    PTB = "PTB"


@dataclass
class Regime:
    """A single contiguous regime interval."""
    type: RegimeType
    t_start: float  # seconds
    t_end: float    # seconds
    original_type: RegimeType | None = None  # pre-PTB classification

    @property
    def duration_s(self) -> float:
        return self.t_end - self.t_start

    @property
    def duration_yr(self) -> float:
        return self.duration_s / YEAR


@dataclass
class RegimeResult:
    """Full regime classification output."""
    regimes: list[Regime]
    t: np.ndarray
    gamma: np.ndarray


def _wrap_to_pi(gamma: np.ndarray) -> np.ndarray:
    """Wrap angle to [-π, π]."""
    return (gamma + np.pi) % (2 * np.pi) - np.pi


def _classify_window(
    gamma_window: np.ndarray,
    libration_threshold: float,
) -> RegimeType:
    """Classify a contiguous window of γ values.

    Uses the extrema (max and min) of γ within the window to determine
    if the planet is librating around 0, π, or spinning through.

    Args:
        gamma_window: γ values in the window (radians, unwrapped).
        libration_threshold: Max |γ| amplitude for tidal lock (rad).

    Returns:
        RegimeType classification.
    """
    wrapped = _wrap_to_pi(gamma_window)
    gamma_max = np.max(wrapped)
    gamma_min = np.min(wrapped)

    # Check TL_ZERO: all wrapped values within [-threshold, threshold] of 0
    if gamma_max < libration_threshold and gamma_min > -libration_threshold:
        return RegimeType.TL_ZERO

    # Check TL_PI: shift by π and check
    shifted = _wrap_to_pi(gamma_window - np.pi)
    if np.max(shifted) < libration_threshold and np.min(shifted) > -libration_threshold:
        return RegimeType.TL_PI

    # Otherwise spinning (monotonic drift through angles)
    return RegimeType.SPINNING


def _find_regime_boundaries(
    t: np.ndarray,
    gamma: np.ndarray,
    window_orbits: int = 50,
) -> list[tuple[int, int, RegimeType]]:
    """Identify contiguous regime segments by sliding window classification.

    Uses overlapping windows of `window_orbits` samples, classifying each
    window and merging adjacent windows with the same classification.

    Args:
        t: Time array (seconds).
        gamma: Spin-orbit angle array (radians).
        window_orbits: Window size in samples for classification.

    Returns:
        List of (start_idx, end_idx, RegimeType) tuples.
    """
    n = len(t)
    if n < window_orbits:
        regime_type = _classify_window(gamma, 2.0)
        return [(0, n - 1, regime_type)]

    # Classify each window
    step = max(1, window_orbits // 4)
    window_starts = list(range(0, n - window_orbits, step))
    if not window_starts:
        window_starts = [0]

    classifications: list[tuple[int, int, RegimeType]] = []
    for start in window_starts:
        end = min(start + window_orbits, n)
        regime_type = _classify_window(gamma[start:end], 2.0)
        classifications.append((start, end - 1, regime_type))

    # Merge adjacent windows with same type
    if not classifications:
        return [(0, n - 1, RegimeType.SPINNING)]

    merged: list[tuple[int, int, RegimeType]] = [classifications[0]]
    for start, end, rtype in classifications[1:]:
        prev_start, prev_end, prev_type = merged[-1]
        if rtype == prev_type:
            merged[-1] = (prev_start, end, prev_type)
        else:
            merged.append((start, end, rtype))

    return merged


def classify_regimes(
    t: np.ndarray,
    gamma: np.ndarray,
    min_regime_yr: float = 10.0,
    libration_threshold: float = 2.0,
    window_orbits: int = 50,
) -> RegimeResult:
    """Classify spin-orbit time series into regimes.

    Args:
        t: Time array (seconds).
        gamma: Spin-orbit angle γ (radians).
        min_regime_yr: Minimum regime duration (years). Shorter = PTB.
        libration_threshold: Max |γ| for tidal lock (rad). Default 2.0.
        window_orbits: Sliding window size for classification.

    Returns:
        RegimeResult with classified regimes.
    """
    boundaries = _find_regime_boundaries(t, gamma, window_orbits)

    regimes: list[Regime] = []
    for start_idx, end_idx, rtype in boundaries:
        t_start = t[start_idx]
        t_end = t[end_idx]

        # Reclassify short regimes as PTB, preserving original type
        if Regime(type=rtype, t_start=t_start, t_end=t_end).duration_yr < min_regime_yr and rtype != RegimeType.PTB:
            regime = Regime(
                type=RegimeType.PTB, t_start=t_start, t_end=t_end,
                original_type=rtype,
            )
        else:
            regime = Regime(type=rtype, t_start=t_start, t_end=t_end)

        regimes.append(regime)

    return RegimeResult(regimes=regimes, t=t, gamma=gamma)


def compute_regime_stats(result: RegimeResult) -> dict[str, dict[str, float]]:
    """Compute duration statistics for each regime type.

    Args:
        result: RegimeResult from classify_regimes.

    Returns:
        Dict mapping regime type name to statistics dict with keys:
        count, total_yr, mean_yr, median_yr, std_yr, q25_yr, q75_yr.
    """
    stats: dict[str, dict[str, float]] = {}

    for rtype in RegimeType:
        durations = [r.duration_yr for r in result.regimes if r.type == rtype]
        if not durations:
            stats[rtype.value] = {
                "count": 0, "total_yr": 0.0, "mean_yr": 0.0,
                "median_yr": 0.0, "std_yr": 0.0, "q25_yr": 0.0, "q75_yr": 0.0,
            }
            continue

        arr = np.array(durations)
        stats[rtype.value] = {
            "count": len(durations),
            "total_yr": float(np.sum(arr)),
            "mean_yr": float(np.mean(arr)),
            "median_yr": float(np.median(arr)),
            "std_yr": float(np.std(arr)),
            "q25_yr": float(np.percentile(arr, 25)),
            "q75_yr": float(np.percentile(arr, 75)),
        }

    return stats


def compute_regime_fractions(result: RegimeResult) -> dict[str, float]:
    """Compute fraction of total time in each regime type.

    Args:
        result: RegimeResult from classify_regimes.

    Returns:
        Dict mapping regime type name to time fraction (0–1).
    """
    total_time = sum(r.duration_s for r in result.regimes)
    if total_time == 0:
        return {rtype.value: 0.0 for rtype in RegimeType}

    fractions: dict[str, float] = {}
    for rtype in RegimeType:
        type_time = sum(r.duration_s for r in result.regimes if r.type == rtype)
        fractions[rtype.value] = type_time / total_time

    return fractions


def compute_quasi_stable_fraction(
    result: RegimeResult,
    threshold_yr: float = 900.0,
) -> float:
    """Fraction of time spent in regimes longer than threshold.

    A "quasi-stable" regime is one lasting longer than threshold_yr,
    indicating the planet maintains a consistent spin state long enough
    for climate/habitability implications.

    Args:
        result: RegimeResult from classify_regimes.
        threshold_yr: Minimum duration to count as quasi-stable (years).

    Returns:
        Fraction of total time in quasi-stable regimes (0–1).
    """
    total_time = sum(r.duration_s for r in result.regimes)
    if total_time == 0:
        return 0.0

    stable_time = sum(
        r.duration_s for r in result.regimes if r.duration_yr >= threshold_yr
    )
    return stable_time / total_time


def extract_ptb_intervals(result: RegimeResult) -> list[dict]:
    """Find and describe all PTB episodes in the regime sequence.

    Args:
        result: RegimeResult from classify_regimes.

    Returns:
        List of dicts with keys: t_start, t_end, duration_yr,
        underlying_type, index, neighbors.
    """
    intervals: list[dict] = []
    regimes = result.regimes

    for i, r in enumerate(regimes):
        if r.type != RegimeType.PTB:
            continue

        prev_type = regimes[i - 1].type.value if i > 0 else None
        next_type = regimes[i + 1].type.value if i < len(regimes) - 1 else None

        underlying = r.original_type.value if r.original_type is not None else "UNKNOWN"

        intervals.append({
            "t_start": r.t_start,
            "t_end": r.t_end,
            "duration_yr": r.duration_yr,
            "underlying_type": underlying,
            "index": i,
            "neighbors": {"before": prev_type, "after": next_type},
        })

    return intervals
