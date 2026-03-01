#!/usr/bin/env python3
"""
Planet 5 (f) triaxiality resonance investigation.

The PTB sweep found a non-monotonic anomaly at triax=3e-4: PTB increases to
51.7% (from 18% at 1e-4 and 0% at 5e-4). This script investigates whether
this is a resonance between the triaxial libration frequency and a perturbation
forcing frequency.

Three analyses:
1. Fine triax grid sweep around the anomaly
2. Frequency analysis at the resonance point (power spectrum of γ(t) and n(t))
3. Theoretical libration period vs triax curve with forcing period overlay

Usage:
    uv run python tlsr-spin/triax_resonance.py
    uv run python tlsr-spin/triax_resonance.py --quick
    uv run python tlsr-spin/triax_resonance.py --plot
    uv run python tlsr-spin/triax_resonance.py --plot --parallel 16
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from shared.constants import AU, YEAR
from shared.paths import output_dir_for

from tlsr_spin.nbody import build_trappist1_system, integrate_and_extract
from tlsr_spin.physics import eccentricity_function, omega_s_squared
from tlsr_spin.regime_classifier import classify_regimes, compute_regime_fractions
from tlsr_spin.spin_integrator import integrate_spin
from tlsr_spin.validate import TRAPPIST1_PLANETS, TRAPPIST1_STAR_MASS

# Fine triax grid around the anomaly
TRIAX_GRID_FULL = [
    1.5e-4, 2e-4, 2.5e-4, 2.8e-4, 3e-4, 3.2e-4, 3.5e-4, 4e-4, 4.5e-4, 5e-4,
]
TRIAX_GRID_QUICK = [2e-4, 3e-4, 4e-4]

PLANET_IDX = 5
TIDAL_Q = 10


def _run_single_triax(
    triax: float,
    times: np.ndarray,
    e_t: np.ndarray,
    n_t: np.ndarray,
    m_star: float,
    m_planet: float,
    r_planet: float,
    a_mean: float,
    tidal_q: float,
) -> dict:
    """Run spin integration + classification for a single triaxiality value.

    Top-level function (picklable) for use with ProcessPoolExecutor.

    Args:
        triax: (B-A)/C triaxiality.
        times: Time array (seconds).
        e_t: Eccentricity array.
        n_t: Mean motion array (rad/s).
        m_star: Stellar mass (kg).
        m_planet: Planet mass (kg).
        r_planet: Planet radius (m).
        a_mean: Mean semi-major axis (m).
        tidal_q: Tidal quality factor.

    Returns:
        Dict with triax, fractions, spin, regime_result.
    """
    spin = integrate_spin(
        times=times,
        e_t=e_t,
        n_t=n_t,
        m_star=m_star,
        m_planet=m_planet,
        r_planet=r_planet,
        a_mean=a_mean,
        tidal_q=tidal_q,
        triaxiality=triax,
    )

    regime_result = classify_regimes(spin["t"], spin["gamma"])
    fractions = compute_regime_fractions(regime_result)

    return {
        "triax": triax,
        "fractions": fractions,
        "spin": spin,
        "regime_result": regime_result,
    }


def run_triax_sweep(
    n_orbits: int,
    triax_values: list[float],
    parallel: int = 1,
) -> list[dict]:
    """Run spin integration for each triaxiality value.

    Builds a single N-body integration and re-runs the spin ODE with
    different triaxiality values, since triaxiality only enters the
    spin equation (not the N-body).

    Args:
        n_orbits: Number of orbits to integrate.
        triax_values: List of (B-A)/C values to sweep.
        parallel: Number of parallel workers (1 = serial).

    Returns:
        List of result dicts with triax, fractions, spin data, nbody.
    """
    planet = TRAPPIST1_PLANETS[PLANET_IDX]

    print(f"Building TRAPPIST-1 system...")
    sim = build_trappist1_system()

    # Report progress every 10% of the integration
    progress_interval = max(n_orbits // 10, 1000)

    print(f"Integrating {n_orbits:,} orbits for planet {PLANET_IDX} ({planet['name']})...")
    nbody = integrate_and_extract(
        sim, PLANET_IDX, n_orbits, progress_interval=progress_interval,
    )
    print(f"  Duration: {nbody['t'][-1] / YEAR:.0f} years, {len(nbody['t'])} samples")
    print(f"  e range: [{nbody['e'].min():.6f}, {nbody['e'].max():.6f}]")

    a_mean = float(np.mean(nbody["a"])) * AU

    common_args = {
        "times": nbody["t"],
        "e_t": nbody["e"],
        "n_t": nbody["n"],
        "m_star": TRAPPIST1_STAR_MASS,
        "m_planet": planet["mass"],
        "r_planet": planet["radius"],
        "a_mean": a_mean,
        "tidal_q": TIDAL_Q,
    }

    if parallel > 1:
        print(f"\nRunning {len(triax_values)} spin integrations "
              f"across {parallel} workers...")
        results_by_triax: dict[float, dict] = {}

        with ProcessPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(_run_single_triax, triax, **common_args): triax
                for triax in triax_values
            }
            for future in as_completed(futures):
                triax = futures[future]
                result = future.result()
                result["nbody"] = nbody
                results_by_triax[triax] = result

                frac_str = ", ".join(
                    f"{k}={v:.1%}"
                    for k, v in result["fractions"].items()
                    if v > 0.001
                )
                print(f"  triax={triax:.1e}: {frac_str}")

        # Return in original triax order
        results = [results_by_triax[t] for t in triax_values]
    else:
        results = []
        for triax in triax_values:
            print(f"\n  triax={triax:.1e}:")
            result = _run_single_triax(triax, **common_args)
            result["nbody"] = nbody

            frac_str = ", ".join(
                f"{k}={v:.1%}"
                for k, v in result["fractions"].items()
                if v > 0.001
            )
            print(f"    {frac_str}")
            results.append(result)

    return results


def frequency_analysis(
    spin: dict,
    nbody: dict,
) -> dict:
    """Compute power spectra of γ(t) and n(t).

    Args:
        spin: Dict from integrate_spin with t, gamma arrays.
        nbody: Dict from integrate_and_extract with t, n, e arrays.

    Returns:
        Dict with frequency arrays and power spectra.
    """
    # γ(t) power spectrum
    dt_gamma = float(np.median(np.diff(spin["t"])))
    fs_gamma = 1.0 / dt_gamma  # sampling frequency in Hz

    # Use Welch's method for robust spectral estimation
    # nperseg ~ 1/4 of total length for good frequency resolution
    nperseg_gamma = min(len(spin["gamma"]) // 4, 8192)
    if nperseg_gamma < 256:
        nperseg_gamma = len(spin["gamma"]) // 2

    f_gamma, psd_gamma = welch(
        spin["gamma"], fs=fs_gamma, nperseg=nperseg_gamma,
    )

    # n(t) power spectrum — use the N-body time series
    dt_n = float(np.median(np.diff(nbody["t"])))
    fs_n = 1.0 / dt_n

    nperseg_n = min(len(nbody["n"]) // 4, 8192)
    if nperseg_n < 256:
        nperseg_n = len(nbody["n"]) // 2

    # Detrend n(t) to focus on perturbation frequencies
    n_detrended = nbody["n"] - np.mean(nbody["n"])
    f_n, psd_n = welch(n_detrended, fs=fs_n, nperseg=nperseg_n)

    # Convert frequencies from Hz to periods in years
    # f in Hz → T = 1/f in seconds → T/YEAR in years
    with np.errstate(divide="ignore"):
        period_gamma_yr = np.where(f_gamma > 0, 1.0 / (f_gamma * YEAR), np.inf)
        period_n_yr = np.where(f_n > 0, 1.0 / (f_n * YEAR), np.inf)

    return {
        "f_gamma": f_gamma,
        "psd_gamma": psd_gamma,
        "period_gamma_yr": period_gamma_yr,
        "f_n": f_n,
        "psd_n": psd_n,
        "period_n_yr": period_n_yr,
    }


def theoretical_libration_period(
    triax: float,
    n_mean: float,
    e_mean: float,
) -> float:
    """Compute theoretical libration period T_lib = 2π / ω_lib.

    ω_lib = sqrt(ω_s²) = sqrt(3 n² triax |H(e)|)

    Args:
        triax: (B-A)/C triaxiality.
        n_mean: Mean orbital mean motion (rad/s).
        e_mean: Mean eccentricity.

    Returns:
        Libration period in years.
    """
    w_s2 = omega_s_squared(n_mean, triax, e_mean)
    if w_s2 <= 0:
        return np.inf
    t_lib_s = 2.0 * np.pi / np.sqrt(w_s2)
    return t_lib_s / YEAR


def find_dominant_forcing_periods(
    f_n: np.ndarray,
    psd_n: np.ndarray,
    n_peaks: int = 5,
    min_period_yr: float = 0.1,
    max_period_yr: float = 1000.0,
) -> list[dict]:
    """Find dominant peaks in the n(t) power spectrum.

    Args:
        f_n: Frequency array (Hz).
        psd_n: Power spectral density of n(t).
        n_peaks: Number of top peaks to return.
        min_period_yr: Minimum period to consider (years).
        max_period_yr: Maximum period to consider (years).

    Returns:
        List of dicts with period_yr, frequency_hz, power.
    """
    # Convert period bounds to frequency bounds
    f_min = 1.0 / (max_period_yr * YEAR) if max_period_yr < np.inf else 0
    f_max = 1.0 / (min_period_yr * YEAR) if min_period_yr > 0 else np.inf

    mask = (f_n > f_min) & (f_n < f_max) & (f_n > 0)
    if not np.any(mask):
        return []

    f_masked = f_n[mask]
    psd_masked = psd_n[mask]

    # Find top peaks by power
    top_idx = np.argsort(psd_masked)[-n_peaks:][::-1]

    peaks = []
    for idx in top_idx:
        freq = float(f_masked[idx])
        peaks.append({
            "period_yr": 1.0 / (freq * YEAR),
            "frequency_hz": freq,
            "power": float(psd_masked[idx]),
        })

    return peaks


def plot_results(
    sweep_results: list[dict],
    freq_data: dict | None,
    forcing_periods: list[dict],
    n_mean: float,
    e_mean: float,
    out_dir: str,
) -> list[str]:
    """Generate all analysis plots.

    Args:
        sweep_results: Results from run_triax_sweep.
        freq_data: Results from frequency_analysis (or None if quick mode).
        forcing_periods: Dominant forcing periods from n(t).
        n_mean: Mean orbital mean motion (rad/s).
        e_mean: Mean eccentricity.
        out_dir: Output directory for plots.

    Returns:
        List of saved plot paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    saved = []

    # --- Plot 1: PTB fraction vs triaxiality ---
    fig, ax = plt.subplots(figsize=(8, 5))
    triax_vals = [r["triax"] for r in sweep_results]
    ptb_vals = [r["fractions"].get("PTB", 0) * 100 for r in sweep_results]

    ax.plot(triax_vals, ptb_vals, "o-", color="C0", linewidth=2, markersize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Triaxiality (B-A)/C")
    ax.set_ylabel("PTB fraction (%)")
    ax.set_title(f"Planet 5 (f): PTB vs triaxiality (Q={TIDAL_Q})")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)

    path = os.path.join(out_dir, "triax_ptb_fraction.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)
    print(f"  Saved: {path}")

    # --- Plot 2: Libration period vs triax with forcing periods ---
    fig, ax = plt.subplots(figsize=(8, 5))

    triax_range = np.geomspace(1e-5, 1e-2, 200)
    t_lib_yr = [theoretical_libration_period(t, n_mean, e_mean) for t in triax_range]

    ax.plot(triax_range, t_lib_yr, "k-", linewidth=2, label="T_lib (theoretical)")

    # Overlay forcing periods as horizontal lines
    colors = ["C1", "C2", "C3", "C4", "C5"]
    for i, peak in enumerate(forcing_periods[:5]):
        ax.axhline(
            peak["period_yr"],
            color=colors[i % len(colors)],
            linestyle="--",
            linewidth=1.5,
            label=f"Forcing: {peak['period_yr']:.1f} yr",
        )

    # Mark the anomaly triaxiality
    ax.axvline(3e-4, color="red", linestyle=":", linewidth=1, alpha=0.7,
               label="triax=3e-4 (anomaly)")

    # Mark the libration period at the anomaly
    t_lib_anomaly = theoretical_libration_period(3e-4, n_mean, e_mean)
    ax.plot(3e-4, t_lib_anomaly, "r*", markersize=15, zorder=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Triaxiality (B-A)/C")
    ax.set_ylabel("Period (years)")
    ax.set_title("Libration period vs triaxiality with forcing periods")
    ax.set_ylim(0.1, 1000)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    path = os.path.join(out_dir, "triax_libration_vs_forcing.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)
    print(f"  Saved: {path}")

    # --- Plot 3: Power spectra (if available) ---
    if freq_data is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # γ(t) power spectrum
        mask = freq_data["period_gamma_yr"] < 500
        mask &= freq_data["period_gamma_yr"] > 0.1
        ax1.semilogy(
            freq_data["period_gamma_yr"][mask],
            freq_data["psd_gamma"][mask],
            "C0-", linewidth=0.5,
        )
        ax1.set_xlabel("Period (years)")
        ax1.set_ylabel("PSD of γ(t)")
        ax1.set_title(f"γ(t) power spectrum at triax=3e-4 (Q={TIDAL_Q})")
        ax1.set_xscale("log")

        # Mark forcing periods
        for i, peak in enumerate(forcing_periods[:5]):
            ax1.axvline(peak["period_yr"], color=colors[i % len(colors)],
                        linestyle="--", alpha=0.7,
                        label=f"{peak['period_yr']:.1f} yr")

        # Mark theoretical libration period
        t_lib_at_resonance = theoretical_libration_period(3e-4, n_mean, e_mean)
        ax1.axvline(t_lib_at_resonance, color="red", linestyle=":",
                    linewidth=2, label=f"T_lib={t_lib_at_resonance:.1f} yr")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # n(t) power spectrum
        mask_n = freq_data["period_n_yr"] < 500
        mask_n &= freq_data["period_n_yr"] > 0.1
        ax2.semilogy(
            freq_data["period_n_yr"][mask_n],
            freq_data["psd_n"][mask_n],
            "C1-", linewidth=0.5,
        )
        ax2.set_xlabel("Period (years)")
        ax2.set_ylabel("PSD of n(t)")
        ax2.set_title("n(t) perturbation spectrum")
        ax2.set_xscale("log")
        ax2.grid(True, alpha=0.3)

        # Mark dominant peaks
        for i, peak in enumerate(forcing_periods[:5]):
            ax2.axvline(peak["period_yr"], color=colors[i % len(colors)],
                        linestyle="--", alpha=0.7)

        path = os.path.join(out_dir, "triax_power_spectra.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)
        print(f"  Saved: {path}")

    return saved


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Planet 5 triaxiality resonance investigation"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 3 triax values, 100k orbits")
    parser.add_argument("--n-orbits", type=int, default=None,
                        help="Override number of orbits")
    parser.add_argument("--plot", action="store_true",
                        help="Generate plots")
    parser.add_argument("--no-freq", action="store_true",
                        help="Skip frequency analysis")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel workers for spin sweeps (default: 1)")
    args = parser.parse_args()

    out_dir = output_dir_for(__file__)
    os.makedirs(out_dir, exist_ok=True)

    if args.quick:
        triax_grid = TRIAX_GRID_QUICK
        n_orbits = args.n_orbits or 100_000
    else:
        triax_grid = TRIAX_GRID_FULL
        n_orbits = args.n_orbits or 1_000_000

    # Step 1: Fine triax grid sweep
    print("=" * 60)
    print("Step 1: Fine triaxiality grid sweep")
    print("=" * 60)
    sweep_results = run_triax_sweep(n_orbits, triax_grid, parallel=args.parallel)

    # Print summary table
    print("\n" + "=" * 60)
    print("PTB fraction summary:")
    print(f"{'Triax':>10s}  {'TL_ZERO':>8s}  {'TL_PI':>8s}  {'PTB':>8s}  {'SPINNING':>8s}")
    for r in sweep_results:
        f = r["fractions"]
        print(f"{r['triax']:10.1e}  {f.get('TL_ZERO', 0):7.1%}  "
              f"{f.get('TL_PI', 0):7.1%}  {f.get('PTB', 0):7.1%}  "
              f"{f.get('SPINNING', 0):7.1%}")

    # Get mean n and e from the N-body (same for all triax values)
    nbody = sweep_results[0]["nbody"]
    n_mean = float(np.mean(nbody["n"]))
    e_mean = float(np.mean(nbody["e"]))

    print(f"\nMean n = {n_mean:.4e} rad/s")
    print(f"Mean e = {e_mean:.6f}")

    # Print theoretical libration periods
    print("\nTheoretical libration periods:")
    for triax in triax_grid:
        t_lib = theoretical_libration_period(triax, n_mean, e_mean)
        print(f"  triax={triax:.1e}: T_lib = {t_lib:.2f} yr")

    # Step 2: Frequency analysis at the resonance point
    freq_data = None
    forcing_periods: list[dict] = []

    if not args.no_freq:
        print("\n" + "=" * 60)
        print("Step 2: Frequency analysis at triax=3e-4")
        print("=" * 60)

        # Find the triax=3e-4 result (or closest)
        resonance_result = None
        for r in sweep_results:
            if abs(r["triax"] - 3e-4) < 1e-5:
                resonance_result = r
                break

        if resonance_result is None:
            # Run a dedicated integration at 3e-4
            print("  Running dedicated integration at triax=3e-4...")
            dedicated = run_triax_sweep(n_orbits, [3e-4])
            resonance_result = dedicated[0]

        freq_data = frequency_analysis(resonance_result["spin"], nbody)

        # Find dominant forcing periods
        forcing_periods = find_dominant_forcing_periods(
            freq_data["f_n"], freq_data["psd_n"],
        )

        print("\nDominant forcing periods from n(t):")
        for i, peak in enumerate(forcing_periods):
            print(f"  #{i + 1}: T = {peak['period_yr']:.2f} yr "
                  f"(f = {peak['frequency_hz']:.4e} Hz, power = {peak['power']:.2e})")

        # Compare with libration period at resonance
        t_lib_res = theoretical_libration_period(3e-4, n_mean, e_mean)
        print(f"\nTheoretical T_lib at triax=3e-4: {t_lib_res:.2f} yr")

        # Check for commensurabilities
        print("\nCommensurability check (T_forcing / T_lib):")
        for i, peak in enumerate(forcing_periods):
            ratio = peak["period_yr"] / t_lib_res
            # Check for near-integer or simple fraction ratios
            near_int = round(ratio)
            near_half = round(ratio * 2) / 2
            residual_int = abs(ratio - near_int) / max(near_int, 1)
            residual_half = abs(ratio - near_half) / max(near_half, 0.5)

            best_match = ""
            if residual_int < 0.1:
                best_match = f" ≈ {near_int}:1"
            elif residual_half < 0.1:
                best_match = f" ≈ {int(near_half * 2)}:2"

            print(f"  #{i + 1}: {peak['period_yr']:.2f} / {t_lib_res:.2f} = "
                  f"{ratio:.3f}{best_match}")

    # Step 3: Plots
    if args.plot:
        print("\n" + "=" * 60)
        print("Step 3: Generating plots")
        print("=" * 60)

        # If no frequency analysis, still compute forcing periods from
        # a quick spectrum for the plot
        if not forcing_periods:
            from scipy.signal import welch as _welch

            n_det = nbody["n"] - np.mean(nbody["n"])
            dt_n = float(np.median(np.diff(nbody["t"])))
            nperseg = min(len(n_det) // 4, 8192)
            if nperseg < 256:
                nperseg = len(n_det) // 2
            f_n, psd_n = _welch(n_det, fs=1.0 / dt_n, nperseg=nperseg)
            forcing_periods = find_dominant_forcing_periods(f_n, psd_n)

        plot_results(
            sweep_results, freq_data, forcing_periods,
            n_mean, e_mean, out_dir,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
