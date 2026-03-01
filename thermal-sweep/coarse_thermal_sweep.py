"""Coarse thermal sweep across stellar types.

For each stellar type, samples HZ positions and albedos, computes
temperatures using the EBM, and filters for target targets.
"""

import json
import os

from thermal_sweep.ebm import tidal_lock_temperatures
from thermal_sweep.stellar_types import (
    STELLAR_TYPES,
    compute_hz_bounds,
    sample_hz_positions,
)

# Narrative temperature targets (K) - adjusted for liquid water + methane objectives
TEMP_TARGETS = {
    "t_sub": (370, 410),     # Substellar: hot but survivable
    "t_term": (240, 290),    # Terminator: liquid water zone
    "t_anti": (110, 140),    # Antistellar: bulk cold side
}

# Cold trap requirement: some regions must reach 95-110K for liquid methane
# This is achieved through local geometry (craters, shadows), not bulk temperature
COLD_TRAP_TARGET = (95, 110)  # Will be ~30-40K below bulk antistellar

# Fixed parameters for coarse sweep
# Expand albedo range to tune temperatures
COARSE_ALBEDOS = [0.30, 0.35, 0.40, 0.45]  # Lower albedo = warmer
# Expand CO2 range to tune greenhouse effect
COARSE_CO2_FRACTIONS = [0.003, 0.006, 0.010]  # 0.3%, 0.6%, 1.0%
PRESSURE_ATM = 1.0


def score_temperatures(temps: dict) -> tuple[bool, float, str]:
    """Score temperature fit to target targets.

    Args:
        temps: Dict with t_substellar, t_terminator, t_antistellar keys.

    Returns:
        Tuple of (passes_filter, score, reason).
        Score is 0-100 (higher is better). passes_filter is True if all zones
        are within ±15% of target ranges AND cold traps can reach methane range.
    """
    t_sub = temps["t_substellar"]
    t_term = temps["t_terminator"]
    t_anti = temps["t_antistellar"]

    # Check hard filters (±15% tolerance)
    sub_min, sub_max = TEMP_TARGETS["t_sub"]
    term_min, term_max = TEMP_TARGETS["t_term"]
    anti_min, anti_max = TEMP_TARGETS["t_anti"]

    tolerance = 0.15
    sub_pass = (sub_min * (1 - tolerance)) <= t_sub <= (sub_max * (1 + tolerance))
    term_pass = (term_min * (1 - tolerance)) <= t_term <= (term_max * (1 + tolerance))
    anti_pass = (anti_min * (1 - tolerance)) <= t_anti <= (anti_max * (1 + tolerance))

    # Cold trap check: bulk antistellar - 35K should reach liquid methane range
    cold_trap_est = t_anti - 35  # Empirical offset for deep craters
    cold_trap_pass = COLD_TRAP_TARGET[0] <= cold_trap_est <= COLD_TRAP_TARGET[1]

    passes = sub_pass and term_pass and anti_pass and cold_trap_pass

    # Compute score (distance from ideal midpoint)
    sub_ideal = (sub_min + sub_max) / 2
    term_ideal = (term_min + term_max) / 2
    anti_ideal = (anti_min + anti_max) / 2

    sub_err = abs(t_sub - sub_ideal) / sub_ideal
    term_err = abs(t_term - term_ideal) / term_ideal
    anti_err = abs(t_anti - anti_ideal) / anti_ideal

    total_err = (sub_err + term_err + anti_err) / 3
    score = max(0, 100 * (1 - total_err * 2))  # 0-100 scale

    # Bonus for cold trap feasibility
    if cold_trap_pass:
        score += 10  # Reward systems that can support liquid methane

    if not passes:
        reasons = []
        if not sub_pass:
            reasons.append(f"T_sub={t_sub:.0f}K out of range")
        if not term_pass:
            reasons.append(f"T_term={t_term:.0f}K out of range")
        if not anti_pass:
            reasons.append(f"T_anti={t_anti:.0f}K out of range")
        if not cold_trap_pass:
            reasons.append(f"Cold traps={cold_trap_est:.0f}K too warm for CH4")
        reason = "; ".join(reasons)
    else:
        reason = f"pass (cold trap ~{cold_trap_est:.0f}K)"

    return passes, score, reason


def run_coarse_thermal_sweep(output_path: str, verbose: bool = True) -> list[dict]:
    """Run coarse thermal sweep across all stellar types.

    Args:
        output_path: Path to save results JSON.
        verbose: Print progress messages.

    Returns:
        List of candidate dicts with stellar/planetary params and temperatures.
    """
    candidates = []
    n_total = len(STELLAR_TYPES) * 3 * len(COARSE_ALBEDOS) * len(COARSE_CO2_FRACTIONS)
    n_checked = 0

    if verbose:
        print(f"Coarse thermal sweep: {n_total} combinations")
        print(f"  Stellar types: {len(STELLAR_TYPES)}")
        print(f"  Distances per star: 3 (inner/mid/outer HZ)")
        print(f"  Albedos: {COARSE_ALBEDOS}")
        print(f"  CO2 fractions: {[f'{x*100:.1f}%' for x in COARSE_CO2_FRACTIONS]}")
        print()

    for star_name, star_params in STELLAR_TYPES.items():
        hz_inner, hz_outer = compute_hz_bounds(
            star_params["luminosity"], star_params["teff"]
        )
        distances = sample_hz_positions(hz_inner, hz_outer, n=3)

        if verbose:
            print(f"{star_name}: HZ={hz_inner:.4f}-{hz_outer:.4f} AU")

        for distance_au in distances:
            for albedo in COARSE_ALBEDOS:
                for co2_frac in COARSE_CO2_FRACTIONS:
                    n_checked += 1

                    # Compute temperatures
                    temps = tidal_lock_temperatures(
                        distance_au=distance_au,
                        luminosity_lsun=star_params["luminosity"],
                        albedo=albedo,
                        co2_fraction=co2_frac,
                        pressure_atm=PRESSURE_ATM,
                    )

                    passes, score, reason = score_temperatures(temps)

                    candidate = {
                        "star_type": star_name,
                        "star_mass": star_params["mass"],
                        "star_luminosity": star_params["luminosity"],
                        "star_teff": star_params["teff"],
                        "distance_au": distance_au,
                        "albedo": albedo,
                        "co2_fraction": co2_frac,
                        "t_sub": round(temps["t_substellar"], 1),
                        "t_term": round(temps["t_terminator"], 1),
                        "t_anti": round(temps["t_antistellar"], 1),
                        "temp_score": score,
                        "temp_pass": passes,
                        "temp_reason": reason,
                    }

                    if passes:
                        candidates.append(candidate)
                        if verbose:
                            print(f"  ✓ {distance_au:.4f} AU, A={albedo:.2f}, CO2={co2_frac*100:.1f}% → "
                                  f"T_sub={temps['t_substellar']:.0f}K, "
                                  f"T_term={temps['t_terminator']:.0f}K, "
                                  f"T_anti={temps['t_antistellar']:.0f}K "
                                  f"(score={score:.0f}, {reason})")

    if verbose:
        print(f"\nThermal sweep complete: {len(candidates)}/{n_total} passed filters")

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(candidates, f, indent=2, default=str)

    if verbose:
        print(f"Saved: {output_path}")

    return candidates


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Coarse thermal sweep")
    parser.add_argument("--output", default="results/stage1_sweep/thermal_candidates.json",
                        help="Output JSON path")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages")
    args = parser.parse_args()

    run_coarse_thermal_sweep(args.output, verbose=not args.quiet)
