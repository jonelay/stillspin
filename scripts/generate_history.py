#!/usr/bin/env python3
"""Generate a 100,000-year spin-orbit history dataset.

Runs a long-duration simulation of the flip-flop slow bouncer candidate
(Q=8, d=0.0745 AU, triax=3e-5) and outputs a structured JSON dataset.

Usage:
    # Quick test (~5,250 years)
    uv run python scripts/generate_history.py --quick

    # Full run (~100,000 years, 2-3 hours)
    uv run python scripts/generate_history.py
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.constants import (
    AU,
    BIPOLARIS_MASS,
    BIPOLARIS_RADIUS,
    M_EARTH,
    M_SUN,
    R_EARTH,
    STAR_MASS,
    YEAR,
)
from shared.paths import output_dir_for
from shared.scenarios import get_scenario
from tlsr_spin.nbody import build_bipolaris_system, integrate_and_extract
from tlsr_spin.regime_classifier import (
    RegimeResult,
    RegimeType,
    classify_regimes,
    compute_regime_fractions,
    compute_regime_stats,
)
from tlsr_spin.spin_integrator import integrate_spin


# Best flip-flop candidate from Study 11
CONFIG = {
    "q": 8,
    "distance_au": 0.0745,
    "triax": 3e-5,
    "albedo": 0.35,
    "co2_pct": 0.6,
}

# Simulation parameters
ORBITAL_PERIOD_DAYS = 19.18
ORBITAL_PERIOD_YEARS = ORBITAL_PERIOD_DAYS / 365.25

# Full run: ~100K years = ~1.9M orbits
FULL_N_ORBITS = 1_900_000
FULL_TARGET_YEARS = 100_000

# Quick test: ~5,250 years = 100K orbits
QUICK_N_ORBITS = 100_000
QUICK_TARGET_YEARS = 5_250


def run_long_simulation(
    n_orbits: int,
    target_years: float,
    samples_per_orbit: int = 10,
    spin_output_points: int = 100_000,
    progress_interval: int = 100_000,
) -> dict:
    """Run a long spin-orbit simulation.

    Args:
        n_orbits: Number of orbits to integrate.
        target_years: Expected duration in years (for solver tuning).
        samples_per_orbit: Output samples per orbit (10 is required for N-body accuracy).
        spin_output_points: Number of output points for spin integration (default: 100K).
        progress_interval: Print progress every N orbits.

    Returns:
        Dict with regime_result, fractions, stats, spin_result.
    """
    overrides = get_scenario("A")
    overrides["BIPOLARIS_DISTANCE_AU"] = CONFIG["distance_au"]
    overrides["SURFACE_ALBEDO"] = CONFIG["albedo"]
    overrides["ATMO_CO2_FRACTION_MID"] = CONFIG["co2_pct"] / 100

    print(f"Building Bipolaris system (d={CONFIG['distance_au']} AU)...")
    sim = build_bipolaris_system(overrides)

    n_nbody_samples = n_orbits * samples_per_orbit
    print(f"Integrating {n_orbits:,} orbits ({n_orbits * ORBITAL_PERIOD_YEARS:.0f} years)...")
    print(f"  N-body samples: {n_nbody_samples:,} ({samples_per_orbit}/orbit)")
    print(f"  Spin output: {spin_output_points:,} points ({n_nbody_samples/spin_output_points:.1f}× reduction)")

    nbody = integrate_and_extract(
        sim,
        planet_idx=3,  # Bipolaris
        n_orbits=n_orbits,
        samples_per_orbit=samples_per_orbit,
        progress_interval=progress_interval,
    )

    # Planet parameters
    m_star = overrides.get("STAR_MASS", STAR_MASS)
    m_planet = overrides.get("BIPOLARIS_MASS", BIPOLARIS_MASS)
    r_planet = overrides.get("BIPOLARIS_RADIUS", BIPOLARIS_RADIUS)
    a_mean = float(np.mean(nbody["a"])) * AU

    print(f"Spin ODE: triax={CONFIG['triax']:.0e}, Q={CONFIG['q']}...")

    # Use dense_output=True in solver for efficiency
    # The solver computes continuous interpolant, then we sample at desired times
    spin = integrate_spin(
        times=nbody["t"],
        e_t=nbody["e"],
        n_t=nbody["n"],
        m_star=m_star,
        m_planet=m_planet,
        r_planet=r_planet,
        a_mean=a_mean,
        tidal_q=CONFIG["q"],
        triaxiality=CONFIG["triax"],
        n_output=spin_output_points,
        rtol=1e-9,
        atol=1e-12,
    )

    print("Classifying regimes...")
    regime_result = classify_regimes(spin["t"], spin["gamma"])
    fractions = compute_regime_fractions(regime_result)
    stats = compute_regime_stats(regime_result)

    return {
        "regime_result": regime_result,
        "fractions": fractions,
        "stats": stats,
        "spin_result": spin,
        "nbody": nbody,
        "duration_yr": float((nbody["t"][-1] - nbody["t"][0]) / YEAR),
    }


def extract_episodes(regime_result: RegimeResult) -> list[dict]:
    """Extract episode list from regime result.

    Args:
        regime_result: Classified regime result.

    Returns:
        List of episode dicts with type, start_yr, end_yr, duration_yr.
    """
    episodes = []
    regimes = regime_result.regimes

    for i, regime in enumerate(regimes):
        neighbors = None
        if regime.type == RegimeType.PTB:
            neighbors = {
                "before": regimes[i - 1].type.value if i > 0 else None,
                "after": regimes[i + 1].type.value if i < len(regimes) - 1 else None,
            }

        episodes.append({
            "type": regime.type.value,
            "start_yr": round(regime.t_start / YEAR, 2),
            "end_yr": round(regime.t_end / YEAR, 2),
            "duration_yr": round(regime.duration_yr, 2),
            "neighbors": neighbors,
        })

    return episodes


def compute_period_summary(
    regime_result: RegimeResult,
    duration_yr: float,
    period_yr: int = 10,
) -> list[dict]:
    """Compute per-period summary of dominant regime.

    Args:
        regime_result: Classified regime result.
        duration_yr: Total simulation duration in years.
        period_yr: Period length in years (10 for decade, 100 for century).

    Returns:
        List of period summary dicts.
    """
    periods = []
    n_periods = int(duration_yr // period_yr)

    regimes = regime_result.regimes

    for period_idx in range(n_periods):
        period_start_yr = period_idx * period_yr
        period_end_yr = (period_idx + 1) * period_yr
        period_start_s = period_start_yr * YEAR
        period_end_s = period_end_yr * YEAR

        regime_time = {rt.value: 0.0 for rt in RegimeType}

        for regime in regimes:
            overlap_start = max(regime.t_start, period_start_s)
            overlap_end = min(regime.t_end, period_end_s)
            if overlap_start < overlap_end:
                regime_time[regime.type.value] += overlap_end - overlap_start

        total_period_time = sum(regime_time.values())
        if total_period_time == 0:
            continue

        dominant = max(regime_time, key=regime_time.get)
        ptb_frac = regime_time["PTB"] / total_period_time

        periods.append({
            "year": period_idx * period_yr,
            "dominant_regime": dominant,
            "ptb_fraction": round(ptb_frac, 3),
        })

    return periods


def count_flips(episodes: list[dict]) -> int:
    """Count TL_ZERO <-> TL_PI transitions (flip-flops).

    A flip is defined as a PTB transition between TL_ZERO and TL_PI.

    Args:
        episodes: List of episode dicts.

    Returns:
        Number of flip-flops.
    """
    n_flips = 0
    for ep in episodes:
        if ep["type"] == "PTB" and ep.get("neighbors"):
            before = ep["neighbors"].get("before")
            after = ep["neighbors"].get("after")
            if before and after:
                # Check for TL_ZERO <-> TL_PI transition in either direction
                if (before == "TL_ZERO" and after == "TL_PI") or \
                   (before == "TL_PI" and after == "TL_ZERO"):
                    n_flips += 1
    return n_flips


def main():
    parser = argparse.ArgumentParser(description="Generate 100K-year spin-orbit history")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: ~5,250 years instead of 100,000")
    parser.add_argument("--n-orbits", type=int, default=None,
                        help="Custom number of orbits (overrides --quick)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: results/bipolaris_history_100k.json)")
    parser.add_argument("--samples-per-orbit", type=int, default=10,
                        help="Samples per orbit (default: 10 for accuracy)")
    args = parser.parse_args()

    if args.n_orbits:
        n_orbits = args.n_orbits
        target_years = int(n_orbits * ORBITAL_PERIOD_YEARS)
        suffix = f"{target_years//1000}k"
    elif args.quick:
        n_orbits = QUICK_N_ORBITS
        target_years = QUICK_TARGET_YEARS
        suffix = "5k"
    else:
        n_orbits = FULL_N_ORBITS
        target_years = FULL_TARGET_YEARS
        suffix = "100k"

    print(f"=== Generate {target_years:,}-Year Spin-Orbit History ===")
    print(f"Config: Q={CONFIG['q']}, d={CONFIG['distance_au']} AU, triax={CONFIG['triax']:.0e}")
    print(f"Target: {n_orbits:,} orbits (~{target_years:,} years)")
    print()

    t_start = time.time()

    # Run simulation with optimized spin output
    # Target: ~1 sample/year for regime classification (captures ~1-yr PTB episodes)
    # For 100K years: 19M N-body samples → 100K spin samples (190× reduction)
    # For 5K years: 1M N-body samples → 5K spin samples (200× reduction)
    spin_output = int(target_years * 1.0)  # 1 sample/year
    result = run_long_simulation(
        n_orbits=n_orbits,
        target_years=target_years,
        samples_per_orbit=args.samples_per_orbit,
        spin_output_points=spin_output,
    )

    compute_time = time.time() - t_start
    print(f"\nSimulation complete in {compute_time:.0f}s ({compute_time/3600:.2f} hours)")

    # Extract data
    episodes = extract_episodes(result["regime_result"])
    n_flips = count_flips(episodes)

    # Use century summary for runs > 10K years, decade summary otherwise
    if result["duration_yr"] > 10_000:
        period_summary = compute_period_summary(
            result["regime_result"], result["duration_yr"], period_yr=100
        )
        period_type = "century"
    else:
        period_summary = compute_period_summary(
            result["regime_result"], result["duration_yr"], period_yr=10
        )
        period_type = "decade"

    # Compute TL durations
    tl_durations = [
        ep["duration_yr"]
        for ep in episodes
        if ep["type"] in ("TL_ZERO", "TL_PI")
    ]
    mean_tl = float(np.mean(tl_durations)) if tl_durations else 0
    max_tl = float(np.max(tl_durations)) if tl_durations else 0

    # Build output structure
    output = {
        "metadata": {
            "config": CONFIG,
            "simulation_years": round(result["duration_yr"], 1),
            "n_orbits": n_orbits,
            "samples_per_orbit": args.samples_per_orbit,
            "generated": datetime.now().isoformat(),
            "compute_time_s": round(compute_time, 1),
        },
        "summary": {
            "fractions": {k: round(v, 4) for k, v in result["fractions"].items()},
            "n_episodes": len(episodes),
            "n_flips": n_flips,
            "mean_tl_duration_yr": round(mean_tl, 1),
            "max_tl_duration_yr": round(max_tl, 1),
            "stats": {
                k: {sk: round(sv, 2) for sk, sv in v.items()}
                for k, v in result["stats"].items()
            },
        },
        "episodes": episodes,
        "period_summary": period_summary,
        "period_type": period_type,
    }

    # Output path
    if args.output:
        out_path = args.output
    else:
        out_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results"
        )
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"bipolaris_history_{suffix}.json")

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved: {out_path}")
    print(f"  Size: {os.path.getsize(out_path) / 1024:.1f} KB")

    # Print summary
    print("\n=== Summary ===")
    print(f"Duration: {result['duration_yr']:.0f} years")
    print(f"Episodes: {len(episodes)}")
    print(f"Flip-flops: {n_flips}")
    print(f"Fractions: TL_ZERO={result['fractions'].get('TL_ZERO', 0)*100:.1f}%, "
          f"TL_PI={result['fractions'].get('TL_PI', 0)*100:.1f}%, "
          f"PTB={result['fractions'].get('PTB', 0)*100:.1f}%")
    print(f"TL durations: mean={mean_tl:.1f} yr, max={max_tl:.1f} yr")


if __name__ == "__main__":
    main()
