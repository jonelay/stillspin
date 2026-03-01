"""
Period measurement within spin-orbit regimes.

Reimplements DayNite's calcPeriod.py: measures libration and spin periods
using zero-crossing detection within each classified regime window.

For tidally locked regimes (TL_ZERO, TL_PI), the "period" is the
libration period — how long it takes γ to complete one oscillation
around the equilibrium. For SPINNING regimes, it's the time for γ
to advance by 2π.
"""

import numpy as np

from shared.constants import YEAR
from .regime_classifier import Regime, RegimeResult, RegimeType


def _find_zero_crossings(
    t: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Find times where y(t) crosses zero using linear interpolation.

    Args:
        t: Time array.
        y: Signal array (same length as t).

    Returns:
        Array of interpolated zero-crossing times.
    """
    sign_changes = np.where(np.diff(np.sign(y)))[0]
    if len(sign_changes) == 0:
        return np.array([])

    crossings = np.empty(len(sign_changes))
    for i, idx in enumerate(sign_changes):
        # Linear interpolation between t[idx] and t[idx+1]
        dy = y[idx + 1] - y[idx]
        if abs(dy) < 1e-30:
            crossings[i] = t[idx]
        else:
            crossings[i] = t[idx] - y[idx] * (t[idx + 1] - t[idx]) / dy

    return crossings


def _measure_libration_period(
    t: np.ndarray,
    gamma: np.ndarray,
    center: float,
) -> np.ndarray:
    """Measure libration periods around a given center (0 or π).

    Period = time between consecutive zero-crossings of (γ - center)
    in the same direction (i.e., full cycle = 2 half-cycles).

    Args:
        t: Time array (seconds).
        gamma: γ array (radians).
        center: Libration center (0 or π).

    Returns:
        Array of measured periods (seconds).
    """
    deviation = gamma - center
    # For TL_PI, wrap so we measure deviation from ±π correctly
    if abs(center) > 1.0:
        deviation = (deviation + np.pi) % (2 * np.pi) - np.pi

    crossings = _find_zero_crossings(t, deviation)
    if len(crossings) < 3:
        return np.array([])

    # Full period = time between every other zero-crossing
    # (each zero-crossing alternates direction)
    periods = np.diff(crossings[::2])
    return periods[periods > 0]


def _measure_spin_period(
    t: np.ndarray,
    gamma: np.ndarray,
) -> np.ndarray:
    """Measure spin period for a spinning (non-locked) regime.

    Period = time for γ to advance by 2π.

    Args:
        t: Time array (seconds).
        gamma: γ array (radians).

    Returns:
        Array of measured periods (seconds).
    """
    # Unwrap γ and measure time per 2π advance
    unwrapped = np.unwrap(gamma)
    total_advance = unwrapped[-1] - unwrapped[0]
    total_time = t[-1] - t[0]

    if abs(total_advance) < 2 * np.pi or total_time == 0:
        return np.array([])

    # Count full rotations
    n_rotations = int(abs(total_advance) / (2 * np.pi))
    if n_rotations < 1:
        return np.array([])

    # Find times where unwrapped γ crosses each 2π boundary
    gamma_shifted = unwrapped - unwrapped[0]
    if total_advance < 0:
        gamma_shifted = -gamma_shifted

    periods = []
    for k in range(1, n_rotations + 1):
        target = k * 2 * np.pi
        # Find crossing of target
        crossings = np.where(np.diff(np.sign(gamma_shifted - target)))[0]
        if len(crossings) == 0:
            continue
        idx = crossings[0]
        dy = gamma_shifted[idx + 1] - gamma_shifted[idx]
        if abs(dy) < 1e-30:
            t_cross = t[idx]
        else:
            t_cross = t[idx] + (target - gamma_shifted[idx]) * (t[idx + 1] - t[idx]) / dy
        periods.append(t_cross)

    if len(periods) < 2:
        return np.array([])

    return np.diff(np.array(periods))


def measure_periods(
    t: np.ndarray,
    gamma: np.ndarray,
    result: RegimeResult,
) -> dict[str, np.ndarray]:
    """Measure periods within each regime.

    Args:
        t: Full time array (seconds).
        gamma: Full γ array (radians).
        result: RegimeResult from classify_regimes.

    Returns:
        Dict mapping regime type name to array of periods (years).
    """
    periods: dict[str, list[float]] = {rtype.value: [] for rtype in RegimeType}

    for regime in result.regimes:
        mask = (t >= regime.t_start) & (t <= regime.t_end)
        if np.sum(mask) < 10:
            continue

        t_window = t[mask]
        g_window = gamma[mask]

        if regime.type == RegimeType.TL_ZERO:
            p = _measure_libration_period(t_window, g_window, center=0.0)
        elif regime.type == RegimeType.TL_PI:
            p = _measure_libration_period(t_window, g_window, center=np.pi)
        elif regime.type == RegimeType.SPINNING:
            p = _measure_spin_period(t_window, g_window)
        else:
            # PTB — try both, take what we can get
            p = _measure_libration_period(t_window, g_window, center=0.0)
            if len(p) == 0:
                p = _measure_spin_period(t_window, g_window)

        if len(p) > 0:
            periods[regime.type.value].extend((p / YEAR).tolist())

    return {k: np.array(v) for k, v in periods.items()}


def period_statistics(periods: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Compute period statistics for each regime type.

    Args:
        periods: Dict from measure_periods (regime name → period array in years).

    Returns:
        Dict mapping regime type name to stats dict with:
        count, mean_yr, median_yr, std_yr, q25_yr, q75_yr.
    """
    stats: dict[str, dict[str, float]] = {}

    for rtype_name, p_arr in periods.items():
        if len(p_arr) == 0:
            stats[rtype_name] = {
                "count": 0, "mean_yr": 0.0, "median_yr": 0.0,
                "std_yr": 0.0, "q25_yr": 0.0, "q75_yr": 0.0,
            }
            continue

        stats[rtype_name] = {
            "count": len(p_arr),
            "mean_yr": float(np.mean(p_arr)),
            "median_yr": float(np.median(p_arr)),
            "std_yr": float(np.std(p_arr)),
            "q25_yr": float(np.percentile(p_arr, 25)),
            "q75_yr": float(np.percentile(p_arr, 75)),
        }

    return stats
