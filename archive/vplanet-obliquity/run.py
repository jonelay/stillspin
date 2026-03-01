#!/usr/bin/env python3
"""
VPLanet obliquity evolution simulation for Bipolaris.

Generates .in files from shared constants, runs VPLanet, and analyzes
the resulting obliquity oscillation.
"""

import argparse
import os
import subprocess
import sys

import numpy as np

from shared.constants import (
    STAR_MASS_MSUN, STAR_LUMINOSITY_LSUN, STAR_TEFF,
    BIPOLARIS_MASS, BIPOLARIS_RADIUS, BIPOLARIS_DISTANCE_AU,
    BIPOLARIS_OBLIQUITY_INITIAL_DEG, BIPOLARIS_ROTATION_PERIOD,
    BIPOLARIS_PERIOD_DAYS,
    NEXARA_MASS, NEXARA_DISTANCE_PLANETARY_RADII, NEXARA_PERIOD,
    M_EARTH, R_EARTH, DAY, YEAR,
    orbital_period_days,
)
from shared.scenarios import get_scenario, get_scenario_label
from shared.plotting import apply_style, save_and_show
from shared.paths import output_dir_for

import matplotlib.pyplot as plt


def generate_in_files(output_dir, t_stop_yr, overrides=None):
    """Generate VPLanet .in files from shared constants."""
    o = overrides or {}
    star_mass = o.get("STAR_MASS_MSUN", STAR_MASS_MSUN)
    star_lum = o.get("STAR_LUMINOSITY_LSUN", STAR_LUMINOSITY_LSUN)
    star_teff = o.get("STAR_TEFF", STAR_TEFF)
    bipolaris_a = o.get("BIPOLARIS_DISTANCE_AU", BIPOLARIS_DISTANCE_AU)

    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    os.makedirs(templates_dir, exist_ok=True)

    # System file
    system_in = f"""# Bipolaris System — VPLanet input
# Auto-generated from shared/constants.py

sName          system
saBodyFiles    star.in bipolaris.in

bDoLog         1
iVerbose       5
bOverwrite     1

# Output
dMinValue      1e-10
bDoForward     1
bVarDt         1
dEta           0.01
dStopTime      {t_stop_yr:.0f}
dOutputTime    {max(t_stop_yr / 2000, 1.0):.1f}
sUnitTime      year
sUnitAngle     deg
sUnitLength    AU
sUnitMass      solar
"""

    # Star file
    star_in = f"""# Primary star ({o.get('STAR_SPECTRAL_TYPE', 'M3V')})
sName          star
saModules      stellar

dMass          {star_mass}
dLuminosity    {star_lum}
dTeff          {star_teff:.1f}
dAge           2.5e9
sStellarModel  baraffe
"""

    # Bipolaris file
    rot_period_days = BIPOLARIS_ROTATION_PERIOD / DAY
    bipolaris_in = f"""# Bipolaris
sName          bipolaris
saModules      distrot eqtide

# Orbital
dSemi          {bipolaris_a}
dEcc           0.01
dObliquity     {BIPOLARIS_OBLIQUITY_INITIAL_DEG}
dRotPeriod     {rot_period_days:.4f}
dPrecA         0

# Physical
dMass          {BIPOLARIS_MASS / M_EARTH:.4f}
dRadius        {BIPOLARIS_RADIUS / R_EARTH:.4f}
dRadGyra       0.5
dK2            0.3
dTidalQ        100

# DistRot
bCalcDynEllip  1

# Output
saOutputOrder  Time Obliquity RotPer PrecA
"""

    files = {
        "system.in": system_in,
        "star.in": star_in,
        "bipolaris.in": bipolaris_in,
    }

    for name, content in files.items():
        path = os.path.join(output_dir, name)
        with open(path, "w") as f:
            f.write(content)
        print(f"Generated: {path}")

    return files


def run_vplanet(output_dir):
    """Run VPLanet in the output directory."""
    try:
        result = subprocess.run(
            ["vplanet", "system.in"],
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            print(f"VPLanet stderr:\n{result.stderr}")
            return False
        print("VPLanet completed successfully.")
        return True
    except FileNotFoundError:
        print("VPLanet not found in PATH. Install with: pip install vplanet")
        return False
    except subprocess.TimeoutExpired:
        print("VPLanet timed out (>10 min).")
        return False


def load_vplanet_output(output_dir):
    """Load VPLanet output files."""
    candidates = [
        os.path.join(output_dir, "system.bipolaris.forward"),
        os.path.join(output_dir, "system.bipolaris.Forward"),
    ]
    for path in candidates:
        if os.path.exists(path):
            data = np.loadtxt(path)
            return {
                "time_yr": data[:, 0],
                "obliquity_deg": data[:, 1],
                "rot_period_days": data[:, 2],
                "prec_a_deg": data[:, 3],
            }
    print("VPLanet output file not found.")
    return None


def measure_period(time_yr, obliquity_deg):
    """Estimate obliquity oscillation period from zero-crossings of (obl - mean)."""
    mean_obl = np.mean(obliquity_deg)
    centered = obliquity_deg - mean_obl

    # Find zero crossings (downward)
    crossings = []
    for i in range(1, len(centered)):
        if centered[i - 1] > 0 and centered[i] <= 0:
            t_cross = time_yr[i - 1] + (time_yr[i] - time_yr[i - 1]) * (
                centered[i - 1] / (centered[i - 1] - centered[i])
            )
            crossings.append(t_cross)

    if len(crossings) >= 2:
        periods = np.diff(crossings)
        return np.median(periods), periods
    return None, np.array([])


def plot_results(data, output_dir):
    """Generate obliquity and spin rate plots."""
    apply_style()

    # Obliquity
    fig, ax = plt.subplots()
    ax.plot(data["time_yr"], data["obliquity_deg"])
    ax.axhline(90, color="red", linestyle="--", alpha=0.5, label="90° (flip threshold)")
    ax.set_xlabel("Time (yr)")
    ax.set_ylabel("Obliquity (°)")
    ax.set_title("Bipolaris Obliquity Evolution")
    ax.legend()
    save_and_show(fig, os.path.join(output_dir, "obliquity_evolution.png"))

    # Spin rate
    fig, ax = plt.subplots()
    ax.plot(data["time_yr"], data["rot_period_days"] * 24)
    ax.set_xlabel("Time (yr)")
    ax.set_ylabel("Rotation Period (hours)")
    ax.set_title("Bipolaris Rotation Period Evolution")
    save_and_show(fig, os.path.join(output_dir, "spin_rate.png"))


def write_period_report(output_dir, period, all_periods, scenario_name=None):
    """Write period measurement report."""
    header = "# Obliquity Period Measurement"
    if scenario_name:
        header += f" — Scenario {scenario_name}"
    report = header + "\n\n"

    if period is not None:
        report += f"**Measured period**: {period:.2f} years\n"
        report += f"**Period range**: {all_periods.min():.2f} – {all_periods.max():.2f} years\n"
        report += f"**Number of cycles**: {len(all_periods)}\n\n"
        report += "## Comparison to Lore\n\n"
        report += f"- Lore states: 5–8 years (provisional)\n"
        report += f"- Simulated: {period:.2f} years\n"

        if period > 20:
            report += "\nThe simulated period is longer than the lore value, "
            report += "consistent with the expectation noted in outstanding-issues.md.\n"
            report += "Ecological timescales should be updated from 'seasonal' to 'generational'.\n"
        elif period < 5:
            report += "\nThe simulated period is shorter than expected. "
            report += "Check moon parameters and outer forcing.\n"
    else:
        report += "**No complete oscillation cycles detected.**\n"
        report += "Integration may be too short or obliquity is not oscillating.\n"

    report += "\n## Limitations\n\n"
    report += ("VPLanet's DistRot module uses secular perturbation theory. "
               "The outer gas giant at 3 AU should be well within "
               "the secular regime. If results seem unreasonable, "
               "an N-body approach (REBOUND + spin integration) may be needed.\n")

    path = os.path.join(output_dir, "period_measurement.md")
    with open(path, "w") as f:
        f.write(report)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="VPLanet obliquity simulation")
    parser.add_argument("--quick", action="store_true", help="Short 100 kyr integration")
    parser.add_argument("--scenario", choices=["A", "B", "C"],
                        help="Parameter sweep scenario (A=baseline, B=lighter perturber, C=closer orbit)")
    args = parser.parse_args()

    t_stop = 100_000 if args.quick else 10_000_000

    overrides = None
    scenario_name = args.scenario
    if scenario_name:
        overrides = get_scenario(scenario_name)

    output_dir = output_dir_for(__file__, scenario_name)

    label = f" (scenario {scenario_name})" if scenario_name else ""
    print(f"Generating .in files for {t_stop:,} year integration{label}...")
    generate_in_files(output_dir, t_stop, overrides)

    print("Running VPLanet...")
    success = run_vplanet(output_dir)
    if not success:
        print("VPLanet run failed. .in files have been generated for manual use.")
        sys.exit(1)

    data = load_vplanet_output(output_dir)
    if data is None:
        print("Could not load VPLanet output.")
        sys.exit(1)

    period, all_periods = measure_period(data["time_yr"], data["obliquity_deg"])

    if period is not None:
        print(f"\nMeasured obliquity period: {period:.2f} years")
    else:
        print("\nNo complete oscillation cycles detected.")

    plot_results(data, output_dir)
    write_period_report(output_dir, period, all_periods, scenario_name)


if __name__ == "__main__":
    main()
