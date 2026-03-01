#!/usr/bin/env python3
"""
REBOUND orbital stability simulation for the Bipolaris system.

Builds the hierarchical triple + planetary system and integrates forward,
computing the MEGNO chaos indicator to assess dynamical stability.
"""

import argparse
import json
import os
import sys

import numpy as np

from shared.constants import (
    AU, YEAR, G, M_SUN, M_EARTH,
    STAR_MASS_MSUN, STAR_TEFF, COMPANION_MASS_MSUN,
    COMPANION_DISTANCE_AU,
    PLANET_B_DISTANCE_AU, PLANET_C_DISTANCE_AU,
    BIPOLARIS_DISTANCE_AU, PLANET_D_DISTANCE_AU,
    PLANET_D_ECCENTRICITY, HZ_INNER_AU, HZ_OUTER_AU,
    BIPOLARIS_PERIOD_YR,
    STAR_LUMINOSITY_LSUN,
    PLANET_B_MASS, PLANET_C_MASS, BIPOLARIS_MASS, PLANET_D_MASS,
    orbital_period_years, hz_inner_au, hz_outer_au,
)
from shared.scenarios import get_scenario, get_scenario_label
from shared.plotting import apply_style, save_and_show
from shared.paths import output_dir_for

import rebound
import matplotlib.pyplot as plt


def build_system(overrides=None):
    """Create the REBOUND simulation with all bodies."""
    o = overrides or {}
    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")

    star_mass = o.get("STAR_MASS_MSUN", STAR_MASS_MSUN)
    bipolaris_a = o.get("BIPOLARIS_DISTANCE_AU", BIPOLARIS_DISTANCE_AU)
    planet_b_a = o.get("PLANET_B_DISTANCE_AU", PLANET_B_DISTANCE_AU)
    planet_c_a = o.get("PLANET_C_DISTANCE_AU", PLANET_C_DISTANCE_AU)
    planet_d_a = o.get("PLANET_D_DISTANCE_AU", PLANET_D_DISTANCE_AU)

    # Primary star
    sim.add(m=star_mass)

    # Inner planets (masses in M_sun for REBOUND units)
    sim.add(m=PLANET_B_MASS / M_SUN, a=planet_b_a)       # planet b
    sim.add(m=PLANET_C_MASS / M_SUN, a=planet_c_a)       # planet c
    sim.add(m=BIPOLARIS_MASS / M_SUN, a=bipolaris_a)     # Bipolaris
    sim.add(m=PLANET_D_MASS / M_SUN, a=planet_d_a,
            e=PLANET_D_ECCENTRICITY)                      # planet d

    # Outer companion (K5V, wide binary)
    sim.add(m=COMPANION_MASS_MSUN, a=COMPANION_DISTANCE_AU)

    sim.move_to_com()

    # Integrator setup
    shortest_period = orbital_period_years(planet_b_a,
                                           star_mass * M_SUN)
    sim.integrator = "whfast"
    sim.dt = shortest_period / 20.0

    # Enable MEGNO
    sim.init_megno()

    return sim


def integrate(sim, t_end_yr, n_samples=1000):
    """Integrate and record orbital elements."""
    times = np.linspace(0, t_end_yr, n_samples)
    n_planets = 4  # b, c, Bipolaris, d (indices 1-4)

    semi_major = np.zeros((n_samples, n_planets))
    eccentricities = np.zeros((n_samples, n_planets))
    megno_values = np.zeros(n_samples)

    for i, t in enumerate(times):
        sim.integrate(t)
        megno_values[i] = sim.megno()
        for j in range(n_planets):
            orb = sim.particles[j + 1].orbit()
            semi_major[i, j] = orb.a
            eccentricities[i, j] = orb.e

    return times, semi_major, eccentricities, megno_values


def check_stability(times, semi_major, eccentricities, megno_values, overrides=None):
    """Run sanity checks on the integration results."""
    o = overrides or {}
    issues = []

    # Check MEGNO convergence
    final_megno = megno_values[-1]
    if final_megno > 4.0:
        issues.append(f"MEGNO diverged to {final_megno:.2f} (>4.0 = chaotic)")
    elif final_megno > 2.5:
        issues.append(f"MEGNO elevated at {final_megno:.2f} (marginally stable)")

    # Check for ejections (semi-major axis > 10x initial)
    planet_names = ["b", "c", "Bipolaris", "d"]
    for j in range(4):
        a_initial = semi_major[0, j]
        a_max = semi_major[:, j].max()
        if a_max > 10 * a_initial:
            issues.append(f"Planet {planet_names[j]} possibly ejected: a_max={a_max:.2f} AU")

    # Check Bipolaris in HZ
    l_star = o.get("STAR_LUMINOSITY_LSUN", STAR_LUMINOSITY_LSUN)
    t_eff = o.get("STAR_TEFF", STAR_TEFF)
    hz_in = hz_inner_au(l_star, t_eff)
    hz_out = hz_outer_au(l_star, t_eff)
    bipolaris_a = semi_major[-1, 2]
    in_hz = hz_in <= bipolaris_a <= hz_out
    hz_note = (f"Bipolaris at {bipolaris_a:.3f} AU; "
               f"HZ = {hz_in:.3f}–{hz_out:.3f} AU: "
               f"{'IN HZ' if in_hz else 'OUTSIDE HZ'}")

    return issues, final_megno, hz_note


def plot_results(times, semi_major, eccentricities, megno_values, output_dir):
    """Generate output plots."""
    apply_style()
    planet_names = ["b", "c", "Bipolaris", "d"]

    # Orbital elements
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for j, name in enumerate(planet_names):
        ax1.plot(times, semi_major[:, j], label=name)
        ax2.plot(times, eccentricities[:, j], label=name)
    ax1.set_ylabel("Semi-major axis (AU)")
    ax1.legend()
    ax2.set_ylabel("Eccentricity")
    ax2.set_xlabel("Time (yr)")
    ax2.legend()
    fig.suptitle("Orbital Element Evolution")
    save_and_show(fig, os.path.join(output_dir, "orbits.png"))

    # MEGNO
    fig, ax = plt.subplots()
    ax.plot(times, megno_values)
    ax.axhline(2.0, color="green", linestyle="--", alpha=0.7, label="Quasi-periodic (2.0)")
    ax.set_xlabel("Time (yr)")
    ax.set_ylabel("MEGNO <Y>")
    ax.set_title("MEGNO Chaos Indicator")
    ax.legend()
    save_and_show(fig, os.path.join(output_dir, "megno.png"))


def write_report(output_dir, t_end, issues, final_megno, hz_note, scenario_name=None):
    """Write stability report."""
    status = "STABLE" if not issues else "ISSUES DETECTED"
    header = f"# Stability Report"
    if scenario_name:
        header += f" — Scenario {scenario_name}"
    report = f"""{header}

**Integration**: {t_end:.0f} years
**Final MEGNO**: {final_megno:.4f}
**Status**: {status}

## Habitable Zone

{hz_note}

## Issues

"""
    if issues:
        for issue in issues:
            report += f"- {issue}\n"
    else:
        report += "None. System appears dynamically stable.\n"

    path = os.path.join(output_dir, "stability_report.md")
    with open(path, "w") as f:
        f.write(report)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="REBOUND stability simulation")
    parser.add_argument("--quick", action="store_true", help="Short 1,000 yr integration")
    parser.add_argument("--scenario", choices=["A", "B", "C"],
                        help="Parameter sweep scenario (A=baseline, B=lighter perturber, C=closer orbit)")
    args = parser.parse_args()

    t_end = 1_000 if args.quick else 1_000_000
    n_samples = 500 if args.quick else 2000

    overrides = None
    scenario_name = args.scenario
    if scenario_name:
        overrides = get_scenario(scenario_name)

    output_dir = output_dir_for(__file__, scenario_name)

    print(f"Building system{f' (scenario {scenario_name})' if scenario_name else ''}...")
    sim = build_system(overrides)
    print(f"Integrating for {t_end:,} years ({n_samples} samples)...")
    times, semi_major, eccentricities, megno_values = integrate(sim, t_end, n_samples)

    # Save raw data
    np.savez(os.path.join(output_dir, "orbital_elements.npz"),
             times=times, semi_major=semi_major,
             eccentricities=eccentricities, megno=megno_values)

    issues, final_megno, hz_note = check_stability(
        times, semi_major, eccentricities, megno_values, overrides)

    print(f"\nFinal MEGNO: {final_megno:.4f}")
    print(hz_note)
    if issues:
        print("Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No stability issues detected.")

    plot_results(times, semi_major, eccentricities, megno_values, output_dir)
    write_report(output_dir, t_end, issues, final_megno, hz_note, scenario_name)

    # Write machine-readable results for sweep
    results = {
        "final_megno": float(final_megno),
        "stable": len(issues) == 0,
        "hz_note": hz_note,
        "issues": issues,
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f)

    if issues:
        sys.exit(1)


if __name__ == "__main__":
    main()
