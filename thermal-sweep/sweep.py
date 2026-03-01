#!/usr/bin/env python3
"""
Parameter sweep to find (distance, CO₂, albedo) combinations matching
target temperature requirements.

Target temperatures:
- Substellar: 330-370 K (too hot for surface life)
- Terminator: 210-260 K (habitable band)
- Antistellar: 95-115 K (cold traps, liquid methane possible)

Methane liquefaction requires T < 111.7 K at 1 atm.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from thermal_sweep.ebm import tidal_lock_temperatures
from shared.constants import STAR_LUMINOSITY_LSUN
from shared.paths import output_dir_for


# Target temperature ranges (K)
TARGET_SUBSTELLAR = (330, 370)
TARGET_TERMINATOR = (210, 260)
TARGET_ANTISTELLAR = (95, 115)
METHANE_BOILING_POINT = 111.7  # K at 1 atm


def check_targets(temps: dict) -> dict:
    """Check if temperatures meet target ranges."""
    t_sub = temps["t_substellar"]
    t_term = temps["t_terminator"]
    t_anti = temps["t_antistellar"]

    return {
        "substellar_ok": TARGET_SUBSTELLAR[0] <= t_sub <= TARGET_SUBSTELLAR[1],
        "terminator_ok": TARGET_TERMINATOR[0] <= t_term <= TARGET_TERMINATOR[1],
        "antistellar_ok": TARGET_ANTISTELLAR[0] <= t_anti <= TARGET_ANTISTELLAR[1],
        "methane_possible": t_anti < METHANE_BOILING_POINT,
        "all_ok": (
            TARGET_SUBSTELLAR[0] <= t_sub <= TARGET_SUBSTELLAR[1]
            and TARGET_TERMINATOR[0] <= t_term <= TARGET_TERMINATOR[1]
            and TARGET_ANTISTELLAR[0] <= t_anti <= TARGET_ANTISTELLAR[1]
        ),
    }


def run_sweep(
    distances: list[float],
    co2_fractions: list[float],
    albedos: list[float],
    luminosity_lsun: float = STAR_LUMINOSITY_LSUN,
) -> list[dict]:
    """Run parameter sweep over all combinations."""
    results = []

    for dist, co2, albedo in itertools.product(distances, co2_fractions, albedos):
        temps = tidal_lock_temperatures(
            distance_au=dist,
            luminosity_lsun=luminosity_lsun,
            albedo=albedo,
            co2_fraction=co2,
        )
        checks = check_targets(temps)

        results.append({
            "distance_au": dist,
            "co2_fraction": co2,
            "albedo": albedo,
            "t_substellar": float(temps["t_substellar"]),
            "t_terminator": float(temps["t_terminator"]),
            "t_antistellar": float(temps["t_antistellar"]),
            "flux": float(temps["flux"]),
            "tau": float(temps["tau"]),
            "substellar_ok": bool(checks["substellar_ok"]),
            "terminator_ok": bool(checks["terminator_ok"]),
            "antistellar_ok": bool(checks["antistellar_ok"]),
            "methane_possible": bool(checks["methane_possible"]),
            "all_ok": bool(checks["all_ok"]),
        })

    return results


def filter_valid(results: list[dict]) -> list[dict]:
    """Return only results meeting all target criteria."""
    return [r for r in results if r["all_ok"]]


def print_summary(results: list[dict], valid: list[dict]) -> None:
    """Print sweep summary."""
    print(f"\nSweep complete: {len(results)} combinations tested")
    print(f"Valid combinations (all targets met): {len(valid)}")

    if valid:
        print("\n=== Valid Parameter Sets ===")
        print(f"{'Dist (AU)':<12} {'CO₂ (%)':<10} {'Albedo':<8} "
              f"{'T_sub (K)':<12} {'T_term (K)':<12} {'T_anti (K)':<12} {'CH₄ liq?':<8}")
        print("-" * 82)
        for r in valid:
            print(f"{r['distance_au']:<12.3f} {r['co2_fraction']*100:<10.1f} {r['albedo']:<8.2f} "
                  f"{r['t_substellar']:<12.0f} {r['t_terminator']:<12.0f} {r['t_antistellar']:<12.0f} "
                  f"{'YES' if r['methane_possible'] else 'no':<8}")
    else:
        # Show best near-misses
        print("\nNo valid combinations. Showing best near-misses:")

        # Score by how many targets are met
        scored = []
        for r in results:
            score = sum([
                r["substellar_ok"],
                r["terminator_ok"],
                r["antistellar_ok"],
            ])
            scored.append((score, r))

        scored.sort(key=lambda x: (-x[0], x[1]["t_antistellar"]))

        print(f"\n{'Dist (AU)':<12} {'CO₂ (%)':<10} {'Albedo':<8} "
              f"{'T_sub (K)':<12} {'T_term (K)':<12} {'T_anti (K)':<12} {'Score':<6}")
        print("-" * 78)
        for score, r in scored[:10]:
            print(f"{r['distance_au']:<12.3f} {r['co2_fraction']*100:<10.1f} {r['albedo']:<8.2f} "
                  f"{r['t_substellar']:<12.0f} {r['t_terminator']:<12.0f} {r['t_antistellar']:<12.0f} "
                  f"{score}/3")


def main():
    parser = argparse.ArgumentParser(
        description="Parameter sweep for Bipolaris thermal equilibrium"
    )
    parser.add_argument(
        "--distances", nargs="+", type=float,
        default=[0.055, 0.058, 0.060, 0.063, 0.065, 0.068, 0.070],
        help="Orbital distances to sweep (AU)"
    )
    parser.add_argument(
        "--co2", nargs="+", type=float,
        default=[0.003, 0.005, 0.007, 0.010],
        help="CO₂ fractions to sweep"
    )
    parser.add_argument(
        "--albedos", nargs="+", type=float,
        default=[0.30, 0.33, 0.35, 0.38, 0.40],
        help="Surface albedos to sweep"
    )
    parser.add_argument(
        "--luminosity", type=float, default=STAR_LUMINOSITY_LSUN,
        help=f"Stellar luminosity in L_sun (default: {STAR_LUMINOSITY_LSUN})"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: auto)"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON"
    )
    args = parser.parse_args()

    print("=== Bipolaris Thermal Equilibrium Sweep ===")
    print(f"Target temperatures:")
    print(f"  Substellar:  {TARGET_SUBSTELLAR[0]}-{TARGET_SUBSTELLAR[1]} K")
    print(f"  Terminator:  {TARGET_TERMINATOR[0]}-{TARGET_TERMINATOR[1]} K")
    print(f"  Antistellar: {TARGET_ANTISTELLAR[0]}-{TARGET_ANTISTELLAR[1]} K")
    print(f"  (Methane liquid requires T < {METHANE_BOILING_POINT} K)")
    print(f"\nSweep parameters:")
    print(f"  Distances: {args.distances} AU")
    print(f"  CO₂: {[f'{c*100}%' for c in args.co2]}")
    print(f"  Albedos: {args.albedos}")
    print(f"  L_star: {args.luminosity} L_sun")

    results = run_sweep(
        distances=args.distances,
        co2_fractions=args.co2,
        albedos=args.albedos,
        luminosity_lsun=args.luminosity,
    )
    valid = filter_valid(results)

    print_summary(results, valid)

    # Save results
    output_dir = Path(args.output) if args.output else Path(output_dir_for(__file__))
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.json or True:  # Always save JSON
        with open(output_dir / "sweep_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {output_dir / 'sweep_results.json'}")

        if valid:
            with open(output_dir / "valid_params.json", "w") as f:
                json.dump(valid, f, indent=2)
            print(f"Saved: {output_dir / 'valid_params.json'}")

    return results, valid


if __name__ == "__main__":
    main()
