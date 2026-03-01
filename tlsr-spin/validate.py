#!/usr/bin/env python3
"""
Validation against Shakespeare & Steffen 2023 results.

Loads the paper's SAfinal*.bin REBOUND archives and runs the full
TLSR pipeline for a known configuration, comparing regime statistics
against their published Table D1.

Default validation target: simulation 169, planet 5, Q=10.

Usage:
    uv run python tlsr-spin/validate.py
    uv run python tlsr-spin/validate.py --sim 169 --planet 5 --q 10

Requires the paper's .bin files in tlsr-spin/data/trappist1/.
"""

import argparse
import os
import sys

import numpy as np

from shared.constants import AU, M_EARTH, M_SUN, R_EARTH, YEAR
from shared.paths import output_dir_for

from tlsr_spin.nbody import build_trappist1_system, integrate_and_extract, load_rebound_archive
from tlsr_spin.period_analysis import measure_periods, period_statistics
from tlsr_spin.plots import plot_spin_history
from tlsr_spin.regime_classifier import (
    classify_regimes,
    compute_quasi_stable_fraction,
    compute_regime_fractions,
    compute_regime_stats,
)
from tlsr_spin.spin_integrator import integrate_spin

# Paper's system parameters (TRAPPIST-1-like)
# Shakespeare & Steffen 2023 Table 1
TRAPPIST1_STAR_MASS = 0.0898 * M_SUN  # kg
TRAPPIST1_TRIAXIALITY = 1e-5  # (B-A)/C, hardcoded in paper

# Planet properties from Table 1 (index 5 = planet f)
# Masses and radii from Agol et al. 2021
TRAPPIST1_PLANETS = {
    1: {"name": "b", "mass": 1.374 * M_EARTH, "radius": 1.116 * R_EARTH,
        "a_au": 0.01155},
    2: {"name": "c", "mass": 1.308 * M_EARTH, "radius": 1.097 * R_EARTH,
        "a_au": 0.01582},
    3: {"name": "d", "mass": 0.388 * M_EARTH, "radius": 0.788 * R_EARTH,
        "a_au": 0.02228},
    4: {"name": "e", "mass": 0.692 * M_EARTH, "radius": 0.920 * R_EARTH,
        "a_au": 0.02928},
    5: {"name": "f", "mass": 1.039 * M_EARTH, "radius": 1.045 * R_EARTH,
        "a_au": 0.03853},
    6: {"name": "g", "mass": 1.321 * M_EARTH, "radius": 1.129 * R_EARTH,
        "a_au": 0.04688},
    7: {"name": "h", "mass": 0.326 * M_EARTH, "radius": 0.755 * R_EARTH,
        "a_au": 0.06193},
}

# Expected results from Table D1 (sim 169, planet 5, Q=10)
# These are approximate — the paper reports distributions across
# multiple initial conditions, so we compare order-of-magnitude.
EXPECTED_SIM169_P5_Q10 = {
    "has_tl_zero": True,
    "has_tl_pi": True,
    "has_spinning": True,
    "tl_fraction_range": (0.3, 0.9),  # combined TL_ZERO + TL_PI
}

# Expected results for fresh N-body validation (from paper Fig 4 / Table D3)
EXPECTED_FRESH = {
    # Planet 3 (d), Q=10: all four regimes present (Fig 4, sim 169)
    (3, 10): {
        "has_spinning": True,
        "has_tl_zero": True,
        "has_tl_pi": True,
    },
    # Planet 4 (e), Q=10: tidally locked, no TLSR
    (4, 10): {
        "has_spinning": False,
        "has_tl_zero": True,
    },
    # Planet 4 (e), Q=100: Q-dependence unlocks TLSR
    (4, 100): {
        "has_spinning": True,
    },
}


def validate(
    bin_path: str,
    planet_idx: int = 5,
    tidal_q: int = 10,
    n_orbits: int | None = None,
) -> dict:
    """Run validation pipeline on a paper archive.

    Args:
        bin_path: Path to SAfinal*.bin file.
        planet_idx: Planet index (1-based).
        tidal_q: Tidal quality factor.
        n_orbits: Number of orbits (None = all available).

    Returns:
        Dict with validation results and comparison.
    """
    planet = TRAPPIST1_PLANETS[planet_idx]
    print(f"Validating: planet {planet_idx} ({planet['name']}), Q={tidal_q}")
    print(f"  Archive: {bin_path}")

    # Extract e(t), n(t)
    print("  Loading N-body archive...")
    nbody = load_rebound_archive(
        bin_path, planet_idx, n_orbits=n_orbits, samples_per_orbit=10,
    )
    print(f"  Duration: {nbody['t'][-1] / YEAR:.0f} years, "
          f"{len(nbody['t'])} samples")
    print(f"  e range: [{nbody['e'].min():.6f}, {nbody['e'].max():.6f}]")

    # Spin integration
    print("  Integrating spin ODE...")
    a_mean = float(np.mean(nbody["a"])) * AU
    spin = integrate_spin(
        times=nbody["t"],
        e_t=nbody["e"],
        n_t=nbody["n"],
        m_star=TRAPPIST1_STAR_MASS,
        m_planet=planet["mass"],
        r_planet=planet["radius"],
        a_mean=a_mean,
        tidal_q=tidal_q,
        triaxiality=TRAPPIST1_TRIAXIALITY,
    )
    print(f"  Tidal ε: {spin['tidal_epsilon']:.4e} 1/s")

    # Regime classification
    print("  Classifying regimes...")
    result = classify_regimes(spin["t"], spin["gamma"])
    stats = compute_regime_stats(result)
    fractions = compute_regime_fractions(result)
    qs_frac = compute_quasi_stable_fraction(result)

    # Period analysis
    periods = measure_periods(spin["t"], spin["gamma"], result)
    pstats = period_statistics(periods)

    # Print results
    print("\n  Regime fractions:")
    for name, frac in fractions.items():
        count = int(stats[name]["count"])
        if frac > 0.001 or count > 0:
            print(f"    {name}: {frac:.1%} ({count} regimes)")

    print(f"  Quasi-stable fraction: {qs_frac:.2%}")

    print("\n  Period statistics:")
    for name, ps in pstats.items():
        if ps["count"] > 0:
            print(f"    {name}: mean={ps['mean_yr']:.1f} yr, "
                  f"median={ps['median_yr']:.1f} yr (n={ps['count']})")

    return {
        "planet_idx": planet_idx,
        "planet_name": planet["name"],
        "tidal_q": tidal_q,
        "spin": spin,
        "regime_result": result,
        "stats": stats,
        "fractions": fractions,
        "quasi_stable_fraction": qs_frac,
        "period_stats": pstats,
    }


def validate_fresh(
    planet_idx: int = 3,
    tidal_q: int = 10,
    n_orbits: int = 10_000_000,
    triaxiality: float = 1e-5,
) -> dict:
    """Run validation with a fresh N-body integration (no .bin required).

    Args:
        planet_idx: Planet index (1-based, 1=b through 7=h).
        tidal_q: Tidal quality factor.
        n_orbits: Number of orbits to integrate.
        triaxiality: (B-A)/C triaxiality parameter.

    Returns:
        Dict with validation results.
    """
    planet = TRAPPIST1_PLANETS[planet_idx]
    print(f"Fresh validation: planet {planet_idx} ({planet['name']}), "
          f"Q={tidal_q}, {n_orbits:,} orbits")

    print("  Building TRAPPIST-1 system...")
    sim = build_trappist1_system()

    print(f"  Integrating {n_orbits:,} orbits...")
    nbody = integrate_and_extract(sim, planet_idx, n_orbits)
    print(f"  Duration: {nbody['t'][-1] / YEAR:.0f} years, "
          f"{len(nbody['t'])} samples")
    print(f"  e range: [{nbody['e'].min():.6f}, {nbody['e'].max():.6f}]")

    # Spin integration
    print("  Integrating spin ODE...")
    a_mean = float(np.mean(nbody["a"])) * AU
    spin = integrate_spin(
        times=nbody["t"],
        e_t=nbody["e"],
        n_t=nbody["n"],
        m_star=TRAPPIST1_STAR_MASS,
        m_planet=planet["mass"],
        r_planet=planet["radius"],
        a_mean=a_mean,
        tidal_q=tidal_q,
        triaxiality=triaxiality,
    )
    print(f"  Tidal ε: {spin['tidal_epsilon']:.4e} 1/s")

    # Regime classification
    print("  Classifying regimes...")
    result = classify_regimes(spin["t"], spin["gamma"])
    stats = compute_regime_stats(result)
    fractions = compute_regime_fractions(result)
    qs_frac = compute_quasi_stable_fraction(result)

    # Period analysis
    periods = measure_periods(spin["t"], spin["gamma"], result)
    pstats = period_statistics(periods)

    print("\n  Regime fractions:")
    for name, frac in fractions.items():
        count = int(stats[name]["count"])
        if frac > 0.001 or count > 0:
            print(f"    {name}: {frac:.1%} ({count} regimes)")

    print(f"  Quasi-stable fraction: {qs_frac:.2%}")

    print("\n  Period statistics:")
    for name, ps in pstats.items():
        if ps["count"] > 0:
            print(f"    {name}: mean={ps['mean_yr']:.1f} yr, "
                  f"median={ps['median_yr']:.1f} yr (n={ps['count']})")

    return {
        "planet_idx": planet_idx,
        "planet_name": planet["name"],
        "tidal_q": tidal_q,
        "spin": spin,
        "regime_result": result,
        "stats": stats,
        "fractions": fractions,
        "quasi_stable_fraction": qs_frac,
        "period_stats": pstats,
    }


def check_fresh_expected(results: dict) -> bool:
    """Compare fresh N-body results against expected behavior.

    Args:
        results: Dict from validate_fresh().

    Returns:
        True if results match expected behavior.
    """
    key = (results["planet_idx"], results["tidal_q"])
    expected = EXPECTED_FRESH.get(key)
    if expected is None:
        print(f"\nNo expected results for planet {key[0]}, Q={key[1]}")
        return True

    fractions = results["fractions"]
    passed = True

    print("\nValidation checks:")
    for regime_key, expected_val in expected.items():
        regime = regime_key.replace("has_", "").upper()
        actual = fractions.get(regime, 0) > 0
        status = "PASS" if actual == expected_val else "FAIL"
        if status == "FAIL":
            passed = False
        print(f"  [{status}] {regime_key}: got {actual}, expected {expected_val}")

    return passed


def check_against_expected(results: dict) -> bool:
    """Compare results against expected values from Table D1.

    Args:
        results: Dict from validate().

    Returns:
        True if results are qualitatively consistent with the paper.
    """
    expected = EXPECTED_SIM169_P5_Q10
    fractions = results["fractions"]
    passed = True

    tl_total = fractions.get("TL_ZERO", 0) + fractions.get("TL_PI", 0)

    checks = [
        ("Has TL_ZERO regimes", fractions.get("TL_ZERO", 0) > 0,
         expected["has_tl_zero"]),
        ("Has TL_PI regimes", fractions.get("TL_PI", 0) > 0,
         expected["has_tl_pi"]),
        ("Has SPINNING regimes", fractions.get("SPINNING", 0) > 0,
         expected["has_spinning"]),
        ("TL fraction in range",
         expected["tl_fraction_range"][0] <= tl_total <= expected["tl_fraction_range"][1],
         True),
    ]

    print("\nValidation checks:")
    for name, actual, expected_val in checks:
        status = "PASS" if actual == expected_val else "FAIL"
        if status == "FAIL":
            passed = False
        print(f"  [{status}] {name}: got {actual}, expected {expected_val}")

    return passed


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate TLSR pipeline against Shakespeare & Steffen 2023"
    )
    parser.add_argument("--fresh", action="store_true",
                        help="Build fresh 7-planet N-body (no .bin required)")
    parser.add_argument("--sim", type=int, default=169,
                        help="Simulation number (default: 169)")
    parser.add_argument("--planet", type=int, default=5,
                        help="Planet index, 1-based (default: 5 = planet f)")
    parser.add_argument("--q", type=int, default=10,
                        help="Tidal Q (default: 10)")
    parser.add_argument("--n-orbits", type=int, default=None,
                        help="Number of orbits (default: 10M for fresh, all for archive)")
    parser.add_argument("--triax", type=float, default=1e-5,
                        help="Triaxiality (B-A)/C (default: 1e-5)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate spin history plot")
    args = parser.parse_args()

    if args.fresh:
        n_orbits = args.n_orbits if args.n_orbits is not None else 10_000_000
        results = validate_fresh(
            args.planet, args.q, n_orbits, args.triax,
        )

        if args.plot:
            out_dir = output_dir_for(__file__)
            label = f"fresh_p{args.planet}_Q{args.q}"
            plot_spin_history(
                results["spin"]["t"],
                results["spin"]["gamma"],
                results["regime_result"],
                os.path.join(out_dir, f"validate_{label}.png"),
                title=f"Fresh validation: planet {args.planet} "
                      f"({results['planet_name']}), Q={args.q}",
            )

        passed = check_fresh_expected(results)
        sys.exit(0 if passed else 1)

    # Archive mode (original behavior)
    data_dir = os.path.join(os.path.dirname(__file__), "data", "trappist1")
    bin_path = os.path.join(data_dir, f"SAfinal{args.sim}.bin")

    if not os.path.exists(bin_path):
        print(f"ERROR: Archive not found: {bin_path}")
        print(f"Copy the paper's SAfinal*.bin files to {data_dir}/")
        sys.exit(1)

    results = validate(bin_path, args.planet, args.q, args.n_orbits)

    if args.plot:
        out_dir = output_dir_for(__file__)
        label = f"sim{args.sim}_p{args.planet}_Q{args.q}"
        plot_spin_history(
            results["spin"]["t"],
            results["spin"]["gamma"],
            results["regime_result"],
            os.path.join(out_dir, f"validate_{label}.png"),
            title=f"Validation: {label}",
        )

    if args.sim == 169 and args.planet == 5 and args.q == 10:
        passed = check_against_expected(results)
        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
