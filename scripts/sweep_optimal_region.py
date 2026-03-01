#!/usr/bin/env python3
"""Focused sweep around optimal v3.0 parameters.

Explores the parameter space near the validated optimal configuration
to understand sensitivity and find potential alternatives.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.constants import L_SUN, M_EARTH, M_SUN, R_EARTH
from thermal_sweep.ebm import tidal_lock_temperatures
from tlsr_spin.sweep import run_single

# Base optimal configuration
BASE_CONFIG = {
    "BIPOLARIS_DISTANCE_AU": 0.070,
    "SURFACE_ALBEDO": 0.35,
    "ATMO_CO2_FRACTION_MID": 0.006,
    "STAR_MASS_MSUN": 0.15,
    "STAR_MASS": 0.15 * M_SUN,
    "STAR_LUMINOSITY_LSUN": 0.0022,
    "STAR_LUMINOSITY": 0.0022 * L_SUN,
    "STAR_TEFF": 2900,
    "BIPOLARIS_MASS": 1.0 * M_EARTH,
    "BIPOLARIS_RADIUS": 1.06 * R_EARTH,
}

# Sweep ranges (tight around optimal)
TRIAXIALITIES = [2.5e-5, 3e-5, 3.5e-5, 4e-5]  # Around 3e-5
TIDAL_QS = [5, 8, 10, 15, 20]                 # Around 10
ALBEDOS = [0.30, 0.33, 0.35, 0.38, 0.40]      # Around 0.35
CO2_FRACTIONS = [0.004, 0.005, 0.006, 0.008]  # Around 0.6%


def _run_job(job):
    """Worker function for sweep."""
    config = job["config"]
    triax = job["triax"]
    q = job["q"]
    n_orbits = job["n_orbits"]

    # Compute temperatures for this config
    temps = tidal_lock_temperatures(
        distance_au=config["BIPOLARIS_DISTANCE_AU"],
        luminosity_lsun=config["STAR_LUMINOSITY_LSUN"],
        albedo=config["SURFACE_ALBEDO"],
        co2_fraction=config["ATMO_CO2_FRACTION_MID"],
    )

    # Run TLSR
    t0 = time.time()
    result = run_single(
        scenario_name="A",
        triaxiality=triax,
        tidal_q=q,
        n_orbits=n_orbits,
        cli_overrides=config,
        quiet=True,
    )
    elapsed = time.time() - t0

    return {
        "config": {
            "albedo": config["SURFACE_ALBEDO"],
            "co2_pct": config["ATMO_CO2_FRACTION_MID"] * 100,
            "triax": triax,
            "q": q,
        },
        "temps": {
            "t_sub": round(temps["t_substellar"], 1),
            "t_term": round(temps["t_terminator"], 1),
            "t_anti": round(temps["t_antistellar"], 1),
            "cold_trap": round(temps["t_antistellar"] - 35, 1),
        },
        "ptb": {
            "fractions": result["fractions"],
            "duration_yr": result["duration_yr"],
            "e_range": result["e_range"],
        },
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/optimal_region_sweep/")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--n-orbits", type=int, default=20_000)
    parser.add_argument("--mode", choices=["full", "triax", "atmo"], default="triax",
                        help="full: all params, triax: triax+Q only, atmo: albedo+CO2 only")
    args = parser.parse_args()

    # Build job list based on mode
    jobs = []

    if args.mode == "triax":
        # Just triax + Q sweep (keep albedo/CO2 fixed)
        for triax in TRIAXIALITIES:
            for q in TIDAL_QS:
                jobs.append({
                    "config": BASE_CONFIG.copy(),
                    "triax": triax,
                    "q": q,
                    "n_orbits": args.n_orbits,
                })

    elif args.mode == "atmo":
        # Just albedo + CO2 sweep (keep triax/Q at optimal)
        for albedo in ALBEDOS:
            for co2 in CO2_FRACTIONS:
                config = BASE_CONFIG.copy()
                config["SURFACE_ALBEDO"] = albedo
                config["ATMO_CO2_FRACTION_MID"] = co2
                jobs.append({
                    "config": config,
                    "triax": 3e-5,  # Optimal
                    "q": 10,        # Optimal
                    "n_orbits": args.n_orbits,
                })

    else:  # full
        # All combinations
        for triax in TRIAXIALITIES:
            for q in TIDAL_QS:
                for albedo in ALBEDOS:
                    for co2 in CO2_FRACTIONS:
                        config = BASE_CONFIG.copy()
                        config["SURFACE_ALBEDO"] = albedo
                        config["ATMO_CO2_FRACTION_MID"] = co2
                        jobs.append({
                            "config": config,
                            "triax": triax,
                            "q": q,
                            "n_orbits": args.n_orbits,
                        })

    print(f"Optimal Region Sweep: {len(jobs)} configurations")
    print(f"  Mode: {args.mode}")
    print(f"  N-body orbits: {args.n_orbits:,}")
    print(f"  Workers: {args.workers}")
    print()

    # Execute
    results = []
    t_start = time.time()

    if args.workers <= 1:
        for i, job in enumerate(jobs):
            cfg = job["config"]
            print(f"[{i+1}/{len(jobs)}] A={cfg['SURFACE_ALBEDO']:.2f}, "
                  f"CO2={cfg['ATMO_CO2_FRACTION_MID']*100:.1f}%, "
                  f"triax={job['triax']:.0e}, Q={job['q']}")
            result = _run_job(job)
            ptb = result["ptb"]["fractions"].get("PTB", 0) * 100
            print(f"  → PTB={ptb:.1f}%, T_term={result['temps']['t_term']}K")
            results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_run_job, job): job for job in jobs}
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                cfg = result["config"]
                ptb = result["ptb"]["fractions"].get("PTB", 0) * 100
                print(f"[{i+1}/{len(jobs)}] A={cfg['albedo']:.2f}, "
                      f"CO2={cfg['co2_pct']:.1f}%, triax={cfg['triax']:.0e}, Q={cfg['q']} "
                      f"→ PTB={ptb:.1f}%")
                results.append(result)

    elapsed = time.time() - t_start

    # Sort by PTB fraction descending
    results.sort(key=lambda r: r["ptb"]["fractions"].get("PTB", 0), reverse=True)

    # Save
    os.makedirs(args.output, exist_ok=True)
    with open(f"{args.output}/results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"\nTop 10 by PTB fraction:")
    for i, r in enumerate(results[:10]):
        cfg = r["config"]
        ptb = r["ptb"]["fractions"].get("PTB", 0) * 100
        print(f"  {i+1}. A={cfg['albedo']:.2f}, CO2={cfg['co2_pct']:.1f}%, "
              f"triax={cfg['triax']:.0e}, Q={cfg['q']}: {ptb:.1f}% PTB, "
              f"T_term={r['temps']['t_term']}K")

    print(f"\nResults saved: {args.output}/results.json")


if __name__ == "__main__":
    main()
