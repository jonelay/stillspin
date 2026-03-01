#!/usr/bin/env python3
"""
TidalPy tidal dissipation calculation for Bipolaris-Moon system.

Computes Love numbers, tidal heating, and surface displacement
for comparison with Earth-Moon system.
"""

import argparse
import json
import os

import numpy as np

from shared.constants import (
    BIPOLARIS_MASS, BIPOLARIS_RADIUS, BIPOLARIS_DISTANCE_AU,
    BIPOLARIS_OBLIQUITY_INITIAL_DEG,
    NEXARA_MASS, NEXARA_DISTANCE, NEXARA_PERIOD,
    PLANET_D_ECCENTRICITY,
    M_EARTH, R_EARTH, G, AU,
    STAR_MASS_MSUN, STAR_LUMINOSITY_LSUN, M_SUN,
)
from shared.scenarios import get_scenario, get_scenario_label
from shared.plotting import apply_style, save_and_show
from shared.paths import output_dir_for

import matplotlib.pyplot as plt

try:
    import TidalPy
    from TidalPy.rheology.models import Andrade
    HAS_TIDALPY = True
except ImportError:
    HAS_TIDALPY = False


# Earth-Moon reference values
MOON_MASS = 7.342e22          # kg
MOON_DISTANCE = 3.844e8      # m
EARTH_K2 = 0.299
EARTH_TIDAL_AMPLITUDE = 0.5  # m (body tide)


def analytic_tidal_amplitude(m_satellite, a_orbit, r_body, m_body, k2):
    """
    Equilibrium tidal bulge height (body tide).
    h ≈ k2 * (m_sat / m_body) * (r_body / a_orbit)^3 * r_body
    """
    return k2 * (m_satellite / m_body) * (r_body / a_orbit)**3 * r_body


def analytic_tidal_heating_ecc(k2, q_factor, m_perturber, a_orbit, r_body, n_orbital, ecc):
    """
    Tidal heating rate (W) from eccentricity tides (Peale & Cassen 1978).
    dE/dt = (21/2)(k₂/Q)(G M²_perturber R⁵ n) / a⁶ × e²
    Valid for small eccentricity (e << 1).
    """
    return (21.0 / 2.0) * (k2 / q_factor) * (
        G * m_perturber**2 * r_body**5 * n_orbital) / (a_orbit**6) * ecc**2


def analytic_tidal_heating_obl(k2, q_factor, m_perturber, a_orbit, r_body, n_orbital, obliquity_rad):
    """
    Tidal heating rate (W) from obliquity tides (Wisdom 2008).
    dE/dt = (3/2)(k₂/Q)(G M²_perturber R⁵ n) / a⁶ × sin²(ε)
    Dominates over eccentricity tides when e ≈ 0 and ε is large.
    """
    return (3.0 / 2.0) * (k2 / q_factor) * (
        G * m_perturber**2 * r_body**5 * n_orbital) / (a_orbit**6) * np.sin(obliquity_rad)**2


def run_analytic(overrides=None):
    """Run analytic tidal calculations (no TidalPy dependency)."""
    o = overrides or {}
    k2 = 0.3  # assumed Earth-like

    # For scenario A (0.15 AU), stellar tides on Bipolaris become relevant
    bipolaris_a_au = o.get("BIPOLARIS_DISTANCE_AU", BIPOLARIS_DISTANCE_AU)
    star_mass_msun = o.get("STAR_MASS_MSUN", STAR_MASS_MSUN)
    star_mass_kg = star_mass_msun * M_SUN
    bipolaris_a_m = bipolaris_a_au * AU

    # Lunar tides on Bipolaris
    h_bipolaris = analytic_tidal_amplitude(
        NEXARA_MASS, NEXARA_DISTANCE, BIPOLARIS_RADIUS, BIPOLARIS_MASS, k2
    )

    # Stellar tides on Bipolaris
    h_stellar = analytic_tidal_amplitude(
        star_mass_kg, bipolaris_a_m, BIPOLARIS_RADIUS, BIPOLARIS_MASS, k2
    )

    # Earth-Moon (reference)
    h_earth = analytic_tidal_amplitude(
        MOON_MASS, MOON_DISTANCE, R_EARTH, M_EARTH, EARTH_K2
    )

    # Orbital mean motion
    n_nexara = 2 * np.pi / NEXARA_PERIOD
    n_moon = 2 * np.pi / (27.3 * 86400)

    # Tidal heating (order of magnitude)
    q_factor = 100  # typical rocky body

    # Bipolaris obliquity in radians
    obliquity_rad = np.radians(BIPOLARIS_OBLIQUITY_INITIAL_DEG)

    # Lunar tidal heating on Bipolaris — obliquity-dominated (e ≈ 0)
    # Eccentricity contribution is negligible for circular moon orbit
    nexara_ecc = 0.01  # assumed near-circular
    heating_bip_ecc = analytic_tidal_heating_ecc(
        k2, q_factor, NEXARA_MASS, NEXARA_DISTANCE, BIPOLARIS_RADIUS, n_nexara, nexara_ecc
    )
    heating_bip_obl = analytic_tidal_heating_obl(
        k2, q_factor, NEXARA_MASS, NEXARA_DISTANCE, BIPOLARIS_RADIUS, n_nexara, obliquity_rad
    )
    heating_bipolaris = heating_bip_ecc + heating_bip_obl

    # Earth-Moon reference (e = 0.0549, obliquity = 23.44°)
    moon_ecc = 0.0549
    earth_obl_rad = np.radians(23.44)
    heating_earth = (
        analytic_tidal_heating_ecc(EARTH_K2, q_factor, MOON_MASS, MOON_DISTANCE, R_EARTH, n_moon, moon_ecc)
        + analytic_tidal_heating_obl(EARTH_K2, q_factor, MOON_MASS, MOON_DISTANCE, R_EARTH, n_moon, earth_obl_rad)
    )

    # Stellar tidal heating on Bipolaris
    n_orbital = 2 * np.pi / (np.sqrt(bipolaris_a_m**3 / (G * star_mass_kg)))
    bipolaris_ecc = o.get("PLANET_D_ECCENTRICITY", 0.02)  # Bipolaris own eccentricity, small
    heating_stellar = (
        analytic_tidal_heating_ecc(k2, q_factor, star_mass_kg, bipolaris_a_m, BIPOLARIS_RADIUS, n_orbital, bipolaris_ecc)
        + analytic_tidal_heating_obl(k2, q_factor, star_mass_kg, bipolaris_a_m, BIPOLARIS_RADIUS, n_orbital, obliquity_rad)
    )

    return {
        "k2": k2,
        "h_bipolaris_m": h_bipolaris,
        "h_stellar_m": h_stellar,
        "h_earth_m": h_earth,
        "heating_bipolaris_W": heating_bipolaris,
        "heating_bip_ecc_W": heating_bip_ecc,
        "heating_bip_obl_W": heating_bip_obl,
        "heating_stellar_W": heating_stellar,
        "heating_earth_W": heating_earth,
        "bipolaris_a_au": bipolaris_a_au,
    }


def run_tidalpy():
    """Run full TidalPy calculation if available."""
    if not HAS_TIDALPY:
        print("TidalPy not installed, using analytic approximation.")
        return None

    print("TidalPy full model: not yet implemented (using analytic).")
    return None


def plot_comparison(results, output_dir):
    """Plot tidal amplitude comparison."""
    apply_style()

    fig, ax = plt.subplots(figsize=(8, 5))
    systems = ["Earth-Moon", "Bipolaris-Moon", "Bipolaris-Star"]
    amplitudes = [results["h_earth_m"], results["h_bipolaris_m"], results["h_stellar_m"]]
    colors = ["#4a90d9", "#d94a4a", "#d9a84a"]

    bars = ax.bar(systems, amplitudes, color=colors, width=0.5)
    ax.set_ylabel("Body Tide Amplitude (m)")
    ax.set_title("Tidal Amplitude Comparison")
    ax.set_yscale("log")

    for bar, val in zip(bars, amplitudes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.2e} m", ha="center", va="bottom", fontsize=9)

    save_and_show(fig, os.path.join(output_dir, "tidal_comparison.png"))


def write_reports(results, output_dir, scenario_name=None):
    """Write markdown reports."""
    header_suffix = f" — Scenario {scenario_name}" if scenario_name else ""

    love_report = f"""# Love Numbers{header_suffix}

| Parameter | Value |
|-----------|-------|
| k₂ (assumed) | {results['k2']:.3f} |
| Method | {'Analytic (Earth-like interior)' } |

Note: Full TidalPy layered calculation would refine this value.
Earth reference: k₂ = {EARTH_K2}
"""
    with open(os.path.join(output_dir, "love_numbers.md"), "w") as f:
        f.write(love_report)

    heating_report = f"""# Tidal Heating Rates{header_suffix}

| System | Body Tide (m) | Heating (W) |
|--------|---------------|-------------|
| Earth-Moon | {results['h_earth_m']:.2e} | {results['heating_earth_W']:.2e} |
| Bipolaris-Moon | {results['h_bipolaris_m']:.2e} | {results['heating_bipolaris_W']:.2e} |
| Bipolaris-Star | {results['h_stellar_m']:.2e} | {results['heating_stellar_W']:.2e} |

**Bipolaris at**: {results['bipolaris_a_au']:.2f} AU

### Heating Breakdown (Bipolaris-Moon)

| Component | Heating (W) |
|-----------|-------------|
| Eccentricity tide | {results['heating_bip_ecc_W']:.2e} |
| Obliquity tide (90°) | {results['heating_bip_obl_W']:.2e} |
| **Total** | {results['heating_bipolaris_W']:.2e} |

**Moon/Earth ratio**: {results['h_bipolaris_m'] / results['h_earth_m']:.2e}× amplitude, {results['heating_bipolaris_W'] / results['heating_earth_W']:.2e}× heating

**Stellar/Earth ratio**: {results['h_stellar_m'] / results['h_earth_m']:.2e}× amplitude, {results['heating_stellar_W'] / results['heating_earth_W']:.2e}× heating

## Caveats

- Equilibrium tide model assumes linear elastic response; amplitudes >1 m may exceed this regime
- Obliquity tide formula assumes spin rate ≠ orbital mean motion (valid for Bipolaris's 28-hr rotation)
- All values are order-of-magnitude estimates

## Interpretation

The moon's tidal effect on Bipolaris is {'significant' if results['h_bipolaris_m'] > 0.01 else 'modest'} compared to Earth's lunar tide.
Surface displacement is on the {'meter' if results['h_bipolaris_m'] > 0.1 else 'centimeter' if results['h_bipolaris_m'] > 0.001 else 'sub-millimeter'} scale.

Stellar tidal effect {'dominates over moon' if results['h_stellar_m'] > results['h_bipolaris_m'] else 'is smaller than moon'} at {results['bipolaris_a_au']:.2f} AU.
"""
    with open(os.path.join(output_dir, "heating_rate.md"), "w") as f:
        f.write(heating_report)

    print(f"Saved: {output_dir}/love_numbers.md")
    print(f"Saved: {output_dir}/heating_rate.md")


def main():
    parser = argparse.ArgumentParser(description="Tidal dissipation calculation")
    parser.add_argument("--scenario", choices=["A", "B", "C"],
                        help="Parameter sweep scenario (A=baseline, B=lighter perturber, C=closer orbit)")
    args = parser.parse_args()

    overrides = None
    scenario_name = args.scenario
    if scenario_name:
        overrides = get_scenario(scenario_name)

    output_dir = output_dir_for(__file__, scenario_name)

    label = f" ({get_scenario_label(scenario_name)})" if scenario_name else ""
    print(f"Running tidal calculation{label}...")

    # Try TidalPy first, fall back to analytic
    tidalpy_results = run_tidalpy()
    results = tidalpy_results if tidalpy_results else run_analytic(overrides)

    print(f"\nBody tide amplitude (Moon): {results['h_bipolaris_m']:.6f} m "
          f"(Earth: {results['h_earth_m']:.4f} m)")
    print(f"Body tide amplitude (Star):   {results['h_stellar_m']:.2e} m")
    print(f"Tidal heating (Moon): {results['heating_bipolaris_W']:.2e} W "
          f"(Earth: {results['heating_earth_W']:.2e} W)")

    plot_comparison(results, output_dir)
    write_reports(results, output_dir, scenario_name)

    # Write machine-readable results for sweep
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f)

    return results


if __name__ == "__main__":
    main()
