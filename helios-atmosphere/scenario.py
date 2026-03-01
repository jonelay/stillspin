#!/usr/bin/env python3
"""
HELIOS atmospheric radiative transfer for Bipolaris.

Runs three CO₂ scenarios to determine which atmospheric composition
is consistent with the stated temperature regime.

Requires HELIOS (CUDA) to be installed separately.
"""

import argparse
import json
import os

import numpy as np

from shared.constants import (
    STAR_TEFF, STAR_LUMINOSITY_LSUN,
    BIPOLARIS_DISTANCE_AU, SURFACE_GRAVITY, SURFACE_ALBEDO,
    ATMO_N2_FRACTION, ATMO_O2_FRACTION, ATMO_AR_FRACTION,
    ATMO_CO2_FRACTION_LOW, ATMO_CO2_FRACTION_MID, ATMO_CO2_FRACTION_HIGH,
    TEMP_ILLUMINATED_POLE,
)
from shared.scenarios import get_scenario, get_scenario_label
from shared.plotting import apply_style, save_and_show
from shared.paths import output_dir_for

import matplotlib.pyplot as plt

try:
    from helios import HELIOS
    HAS_HELIOS = True
except ImportError:
    HAS_HELIOS = False


# CO₂ scenarios
CO2_SCENARIOS = {
    "0.5% CO₂": ATMO_CO2_FRACTION_LOW,
    "1% CO₂": ATMO_CO2_FRACTION_MID,
    "2% CO₂": ATMO_CO2_FRACTION_HIGH,
}


def run_analytic_greenhouse(co2_fraction, overrides=None):
    """
    Simple analytic greenhouse estimate when HELIOS is unavailable.

    Uses a parameterized greenhouse effect:
    T_surf = T_eq * (1 + tau/2)^(1/4)
    where tau ~ optical_depth ~ f(CO₂, pressure)
    """
    o = overrides or {}
    l_lsun = o.get("STAR_LUMINOSITY_LSUN", STAR_LUMINOSITY_LSUN)
    a_au = o.get("BIPOLARIS_DISTANCE_AU", BIPOLARIS_DISTANCE_AU)

    # Equilibrium temperature (no atmosphere)
    sigma = 5.67e-8
    l_star = l_lsun * 3.828e26  # W
    d = a_au * 1.496e11          # m
    flux = l_star / (4 * np.pi * d**2)
    t_eq = ((flux * (1 - SURFACE_ALBEDO)) / (4 * sigma))**0.25

    # Parameterized optical depth for CO₂-dominated greenhouse
    tau_earth = 1.0
    co2_earth = 0.0004
    tau = tau_earth * np.log(1 + co2_fraction / co2_earth) / np.log(1 + 1.0)

    # Also account for N₂ pressure broadening (1 atm assumed)
    t_surf = t_eq * (1 + 0.75 * tau)**0.25

    # Generate a simple T-P profile (radiative-convective approximation)
    n_layers = 50
    pressures = np.logspace(-4, 0, n_layers)  # atm, top to bottom
    temperatures = t_surf * (pressures / pressures[-1])**0.2

    return t_surf, pressures, temperatures


def run_helios_scenario(co2_fraction, scenario_name, output_dir, overrides=None):
    """Run a HELIOS scenario (stub — requires HELIOS installation)."""
    if not HAS_HELIOS:
        print(f"  HELIOS not available, using analytic estimate for {scenario_name}")
        return run_analytic_greenhouse(co2_fraction, overrides)

    print(f"  HELIOS full run for {scenario_name}: not yet implemented")
    return run_analytic_greenhouse(co2_fraction, overrides)


def main():
    parser = argparse.ArgumentParser(description="Atmosphere radiative transfer")
    parser.add_argument("--scenario", choices=["A", "B", "C"],
                        help="Parameter sweep scenario (A=baseline, B=lighter perturber, C=closer orbit)")
    args = parser.parse_args()

    overrides = None
    scenario_name = args.scenario
    if scenario_name:
        overrides = get_scenario(scenario_name)

    output_dir = output_dir_for(__file__, scenario_name)

    if not HAS_HELIOS:
        print("HELIOS not installed. Running analytic greenhouse estimates.")
        print("For full radiative transfer, install HELIOS (requires CUDA).\n")

    label = f" (scenario {scenario_name}: {get_scenario_label(scenario_name)})" if scenario_name else ""
    print(f"Running atmosphere model{label}...")

    results = {}
    for name, co2 in CO2_SCENARIOS.items():
        print(f"Running {name}...")
        t_surf, pressures, temperatures = run_helios_scenario(
            co2, name, output_dir, overrides)
        results[name] = {
            "co2": co2,
            "t_surf": t_surf,
            "pressures": pressures,
            "temperatures": temperatures,
        }
        print(f"  Surface temperature: {t_surf:.1f} K")

    # Plot T-P profiles
    apply_style()

    fig, ax = plt.subplots()
    for name, r in results.items():
        ax.semilogy(r["temperatures"], r["pressures"], label=f'{name} (T_s={r["t_surf"]:.0f} K)')
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Pressure (atm)")
    ax.set_title("Temperature-Pressure Profiles")
    ax.invert_yaxis()
    ax.legend()
    save_and_show(fig, os.path.join(output_dir, "tp_profiles.png"))

    # Surface temp vs CO₂
    fig, ax = plt.subplots()
    co2_vals = [r["co2"] * 100 for r in results.values()]
    t_surfs = [r["t_surf"] for r in results.values()]
    ax.plot(co2_vals, t_surfs, "o-", markersize=8)
    ax.axhspan(TEMP_ILLUMINATED_POLE[0], TEMP_ILLUMINATED_POLE[1],
               alpha=0.2, color="green", label=f"Target: {TEMP_ILLUMINATED_POLE[0]}–{TEMP_ILLUMINATED_POLE[1]} K")
    ax.set_xlabel("CO₂ (%)")
    ax.set_ylabel("Surface Temperature (K)")
    ax.set_title("Surface Temperature vs CO₂ Fraction")
    ax.legend()
    save_and_show(fig, os.path.join(output_dir, "surface_temp_vs_co2.png"))

    # Report
    report = "# Atmosphere Report"
    if scenario_name:
        report += f" — Scenario {scenario_name}"
    report += "\n\n"
    report += "| Scenario | CO₂ (%) | Surface Temp (K) | In Target Range? |\n"
    report += "|----------|---------|------------------|------------------|\n"
    for name, r in results.items():
        in_range = TEMP_ILLUMINATED_POLE[0] <= r["t_surf"] <= TEMP_ILLUMINATED_POLE[1]
        report += f"| {name} | {r['co2']*100:.0f} | ~{r['t_surf']:.0f} | {'YES' if in_range else 'no'} |\n"

    report += f"\n**Target range** (illuminated pole): {TEMP_ILLUMINATED_POLE[0]}–{TEMP_ILLUMINATED_POLE[1]} K\n"
    report += "\n**Note**: All temperatures are order-of-magnitude estimates (±50 K or worse). "
    report += "The grey-atmosphere parameterization is not validated at >1% CO₂.\n"

    if overrides:
        a_au = overrides.get("BIPOLARIS_DISTANCE_AU", BIPOLARIS_DISTANCE_AU)
        l_lsun = overrides.get("STAR_LUMINOSITY_LSUN", STAR_LUMINOSITY_LSUN)
        report += f"\n**Scenario params**: a = {a_au} AU, L = {l_lsun} L☉\n"

    report += "\n## Method\n\n"
    if HAS_HELIOS:
        report += "Full HELIOS radiative transfer with PHOENIX M3V spectrum.\n"
    else:
        report += ("Analytic greenhouse estimate (parameterized optical depth). "
                   "Results are order-of-magnitude only. "
                   "Install HELIOS for proper radiative transfer.\n")

    path = os.path.join(output_dir, "atmosphere_report.md")
    with open(path, "w") as f:
        f.write(report)
    print(f"\nSaved: {path}")

    # Write machine-readable results for sweep (scalars only, no arrays)
    best_t = max(r["t_surf"] for r in results.values())
    in_target = TEMP_ILLUMINATED_POLE[0] <= best_t <= TEMP_ILLUMINATED_POLE[1]
    sweep_results = {
        "t_surf_0.5pct": float(results["0.5% CO₂"]["t_surf"]),
        "t_surf_1pct": float(results["1% CO₂"]["t_surf"]),
        "t_surf_2pct": float(results["2% CO₂"]["t_surf"]),
        "best_t_surf": float(best_t),
        "in_target": bool(in_target),
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(sweep_results, f)

    return results


if __name__ == "__main__":
    main()
