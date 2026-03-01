"""Perturbation probe: 5K-year N-body integration to characterize
orbital perturbations on HZ planets.

Loads post-evolution simulation from Pipeline 1, extracts e(t) and n(t),
computes perturbation metrics (RMS, spectrum), and applies a filter
to select systems likely to produce flip-flop behavior.
"""

import time
import warnings

import numpy as np

# Harmless when loading archives read-only (no function pointers needed)
warnings.filterwarnings("ignore", message=".*reset function pointers.*", category=RuntimeWarning)

from shared.constants import YEAR

from .chain_types import PerturbationProbe


def extract_perturbation_metrics(
    t_s: np.ndarray,
    e_t: np.ndarray,
    n_t: np.ndarray,
) -> dict:
    """Compute perturbation metrics from orbital element time series.

    Returns dict with rms_dn_over_n, rms_de, max_e, mean_n_rad_s.
    """
    n_mean = np.mean(n_t)
    dn_over_n = (n_t - n_mean) / n_mean if n_mean > 0 else np.zeros_like(n_t)

    e_mean = np.mean(e_t)
    de = e_t - e_mean

    return {
        "rms_dn_over_n": float(np.sqrt(np.mean(dn_over_n**2))),
        "rms_de": float(np.sqrt(np.mean(de**2))),
        "max_e": float(np.max(e_t)),
        "mean_n_rad_s": float(n_mean),
    }


def compute_spectrum(t_s: np.ndarray, n_t: np.ndarray, n_peaks: int = 5) -> list[float]:
    """FFT of n(t) variations to find dominant perturbation periods.

    Returns top n_peaks periods in years.
    """
    n_detrended = n_t - np.mean(n_t)

    if len(n_detrended) < 10:
        return []

    fft_vals = np.fft.rfft(n_detrended)
    power = np.abs(fft_vals) ** 2

    dt = (t_s[-1] - t_s[0]) / (len(t_s) - 1)
    freqs = np.fft.rfftfreq(len(n_detrended), d=dt)

    # Skip DC component
    power[0] = 0
    if len(power) < 2:
        return []

    # Find top peaks
    peak_indices = np.argsort(power[1:])[-n_peaks:][::-1] + 1
    periods_s = [1.0 / freqs[i] for i in peak_indices if freqs[i] > 0]
    periods_yr = [p / YEAR for p in periods_s]

    return [round(p, 2) for p in periods_yr if p > 0]


def apply_filter(
    rms_dn: float,
    thresholds: dict | None = None,
) -> str:
    """Apply perturbation filter.

    thresholds=None → always PASS (calibration mode).
    thresholds={min: X, max: Y} → reject outside range.
    """
    if thresholds is None:
        return "PASS"

    min_thresh = thresholds.get("min", 0.0)
    max_thresh = thresholds.get("max", float("inf"))

    if rms_dn < min_thresh:
        return "REJECT_WEAK"
    if rms_dn > max_thresh:
        return "REJECT_STRONG"
    return "PASS"


def run_probe(
    sim_archive_path: str,
    hz_planet_idx: int,
    system_id: str,
    n_years: int = 5000,
    thresholds: dict | None = None,
) -> PerturbationProbe:
    """5K-year N-body probe of post-evolution system.

    Loads saved simulation from Pipeline 1 and extracts orbital
    perturbation metrics for the HZ planet.

    Args:
        sim_archive_path: Path to REBOUND simulation file.
        hz_planet_idx: 1-based planet index in simulation.
        system_id: System ID for linking results.
        n_years: Integration duration in years.
        thresholds: Filter thresholds (None = calibration mode).

    Returns:
        PerturbationProbe with metrics and filter verdict.
    """
    import rebound

    t0 = time.time()

    try:
        sim = rebound.Simulation(sim_archive_path)
    except Exception as e:
        return PerturbationProbe(
            system_id=system_id,
            hz_planet_idx=hz_planet_idx,
            rms_dn_over_n=0.0,
            rms_de=0.0,
            max_e=0.0,
            mean_n_rad_s=0.0,
            dominant_periods_yr=[],
            filter_verdict="ERROR",
            elapsed_s=time.time() - t0,
            error_msg=f"Failed to load sim: {e}",
        )

    # Get orbital period of target planet
    orb = sim.particles[hz_planet_idx].orbit()
    p_orb_yr = orb.P
    samples_per_orbit = 10
    n_orbits = int(n_years / p_orb_yr) if p_orb_yr > 0 else 1000
    n_samples = n_orbits * samples_per_orbit

    t_end = sim.t + n_years
    times_yr = np.linspace(sim.t, t_end, n_samples)

    e_arr = np.zeros(n_samples)
    n_arr = np.zeros(n_samples)

    for i, t in enumerate(times_yr):
        sim.integrate(t)
        orb = sim.particles[hz_planet_idx].orbit()
        e_arr[i] = max(orb.e, 0.0)
        if orb.P > 0:
            n_arr[i] = 2.0 * np.pi / (orb.P * YEAR)
        elif i > 0:
            n_arr[i] = n_arr[i - 1]

    times_s = times_yr * YEAR
    metrics = extract_perturbation_metrics(times_s, e_arr, n_arr)
    spectrum = compute_spectrum(times_s, n_arr)
    verdict = apply_filter(metrics["rms_dn_over_n"], thresholds)

    return PerturbationProbe(
        system_id=system_id,
        hz_planet_idx=hz_planet_idx,
        rms_dn_over_n=metrics["rms_dn_over_n"],
        rms_de=metrics["rms_de"],
        max_e=metrics["max_e"],
        mean_n_rad_s=metrics["mean_n_rad_s"],
        dominant_periods_yr=spectrum,
        filter_verdict=verdict,
        elapsed_s=time.time() - t0,
    )
