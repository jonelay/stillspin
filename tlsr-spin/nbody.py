#!/usr/bin/env python3
"""
N-body eccentricity and mean-motion extraction for TLSR spin dynamics.

Two modes of operation:
1. Build a fresh REBOUND simulation from Bipolaris system parameters
   and extract e(t), n(t) for a target planet.
2. Load an existing REBOUND .bin archive (e.g., MacDonald & Dawson 2018
   TRAPPIST-1-like migration simulations used by Shakespeare & Steffen 2023).

Replaces DayNite's simSAX.py.
"""

import argparse
import os

import numpy as np

from shared.constants import (
    AU,
    BIPOLARIS_DISTANCE_AU,
    BIPOLARIS_ECCENTRICITY,
    BIPOLARIS_MASS,
    G,
    M_EARTH,
    M_SUN,
    PLANET_B_DISTANCE_AU,
    PLANET_B_ECCENTRICITY,
    PLANET_B_MASS,
    PLANET_C_DISTANCE_AU,
    PLANET_C_ECCENTRICITY,
    PLANET_C_MASS,
    PLANET_D_DISTANCE_AU,
    PLANET_D_ECCENTRICITY,
    PLANET_D_MASS,
    STAR_MASS_MSUN,
    YEAR,
    orbital_period_years,
)
from shared.paths import output_dir_for


def build_trappist1_system() -> "rebound.Simulation":
    """Build the 7-planet TRAPPIST-1 system in REBOUND.

    Planet parameters from Agol et al. 2021, as used in
    Shakespeare & Steffen 2023 Table 1.

    Returns:
        Configured rebound.Simulation.
    """
    import rebound

    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")

    star_mass = 0.0898  # M_sun
    sim.add(m=star_mass)

    # Planets b–h: (a_au, mass_earth, eccentricity)
    # Eccentricities from MacDonald & Dawson 2018 post-migration values
    # (resonant chain pumping produces e ~ 0.01–0.1, much higher than
    # present-day observed values from Agol et al. 2021)
    planets = [
        (0.01155, 1.374, 0.02),   # b
        (0.01582, 1.308, 0.03),   # c
        (0.02228, 0.388, 0.06),   # d — target for TLSR at Q=10
        (0.02928, 0.692, 0.04),   # e — target for Q-dependence
        (0.03853, 1.039, 0.05),   # f
        (0.04688, 1.321, 0.03),   # g
        (0.06193, 0.326, 0.04),   # h
    ]
    m_earth_msun = M_EARTH / M_SUN
    for a_au, mass_earth, ecc in planets:
        sim.add(m=mass_earth * m_earth_msun, a=a_au, e=ecc)

    sim.move_to_com()

    shortest_period = orbital_period_years(planets[0][0], star_mass * M_SUN)
    sim.integrator = "whfast"
    sim.dt = shortest_period / 20.0

    return sim


def build_bipolaris_system(overrides: dict | None = None) -> "rebound.Simulation":
    """Create REBOUND simulation of the Bipolaris system.

    System Design v2: 4-planet quasi-resonant chain around M5.5V.
    All planets have non-zero eccentricities from the broken Laplace chain.

    Args:
        overrides: Dict of constant overrides (from scenarios).

    Returns:
        Configured rebound.Simulation.
    """
    import rebound

    o = overrides or {}

    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")

    star_mass = o.get("STAR_MASS_MSUN", STAR_MASS_MSUN)
    sim.add(m=star_mass)

    # Inner planets (4-planet quasi-resonant chain)
    planets = [
        ("b", o.get("PLANET_B_DISTANCE_AU", PLANET_B_DISTANCE_AU),
         o.get("PLANET_B_MASS", PLANET_B_MASS) / M_SUN,
         o.get("PLANET_B_ECCENTRICITY", PLANET_B_ECCENTRICITY)),
        ("c", o.get("PLANET_C_DISTANCE_AU", PLANET_C_DISTANCE_AU),
         o.get("PLANET_C_MASS", PLANET_C_MASS) / M_SUN,
         o.get("PLANET_C_ECCENTRICITY", PLANET_C_ECCENTRICITY)),
        ("Bipolaris", o.get("BIPOLARIS_DISTANCE_AU", BIPOLARIS_DISTANCE_AU),
         o.get("BIPOLARIS_MASS", BIPOLARIS_MASS) / M_SUN,
         o.get("BIPOLARIS_ECCENTRICITY", BIPOLARIS_ECCENTRICITY)),
        ("d", o.get("PLANET_D_DISTANCE_AU", PLANET_D_DISTANCE_AU),
         o.get("PLANET_D_MASS", PLANET_D_MASS) / M_SUN,
         o.get("PLANET_D_ECCENTRICITY", PLANET_D_ECCENTRICITY)),
    ]
    for _name, a, m, e in planets:
        sim.add(m=m, a=a, e=e)

    sim.move_to_com()

    # Integrator setup
    shortest_a = min(p[1] for p in planets)
    shortest_period = orbital_period_years(shortest_a, star_mass * M_SUN)
    sim.integrator = "whfast"
    sim.dt = shortest_period / 20.0

    return sim


def integrate_and_extract(
    sim: "rebound.Simulation",
    planet_idx: int,
    n_orbits: int,
    samples_per_orbit: int = 10,
    progress_interval: int = 0,
) -> dict[str, np.ndarray]:
    """Integrate N-body simulation and extract e(t), n(t) for one planet.

    Args:
        sim: REBOUND simulation (already built).
        planet_idx: Index of target planet (1-based, star is 0).
        n_orbits: Number of orbits of the target planet to integrate.
        samples_per_orbit: Output samples per orbital period.
        progress_interval: Print progress every N orbits (0 = no progress).

    Returns:
        Dict with 't' (seconds), 'e' (eccentricity), 'n' (mean motion rad/s),
        'a' (semi-major axis AU).
    """
    import sys
    import time

    # Get initial orbital period
    orb = sim.particles[planet_idx].orbit()
    p_orb_yr = orb.P  # in simulation units (years)

    t_end = n_orbits * p_orb_yr
    n_samples = n_orbits * samples_per_orbit

    times_yr = np.linspace(0, t_end, n_samples)

    e_arr = np.zeros(n_samples)
    n_arr = np.zeros(n_samples)
    a_arr = np.zeros(n_samples)

    # Progress tracking
    next_progress = progress_interval if progress_interval > 0 else 0
    t_wall_start = time.monotonic()

    for i, t in enumerate(times_yr):
        sim.integrate(t)
        orb = sim.particles[planet_idx].orbit()
        e_arr[i] = max(orb.e, 0.0)  # clip numerical noise
        a_arr[i] = orb.a
        # Mean motion: n = 2π/P, P in years → convert to rad/s
        if orb.P > 0:
            n_arr[i] = 2.0 * np.pi / (orb.P * YEAR)
        else:
            n_arr[i] = n_arr[max(0, i - 1)]

        if next_progress > 0 and (i + 1) % (next_progress * samples_per_orbit) == 0:
            orbits_done = (i + 1) // samples_per_orbit
            pct = orbits_done / n_orbits * 100
            elapsed = time.monotonic() - t_wall_start
            rate = orbits_done / elapsed if elapsed > 0 else 0
            eta = (n_orbits - orbits_done) / rate if rate > 0 else 0
            print(f"  N-body: {orbits_done:,}/{n_orbits:,} orbits "
                  f"({pct:.0f}%) — {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining",
                  flush=True)

    return {
        "t": times_yr * YEAR,  # convert to seconds
        "e": e_arr,
        "n": n_arr,
        "a": a_arr,
    }


def load_rebound_archive(
    bin_path: str,
    planet_idx: int,
    n_orbits: int | None = None,
    samples_per_orbit: int = 10,
) -> dict[str, np.ndarray]:
    """Load a REBOUND .bin archive and extract e(t), n(t).

    The paper's SAfinal*.bin files contain full simulation snapshots
    from MacDonald & Dawson 2018 TRAPPIST-1-like migration runs.

    Args:
        bin_path: Path to .bin archive file.
        planet_idx: Planet index (1-based).
        n_orbits: Number of orbits to extract. None = all available.
        samples_per_orbit: Output samples per orbital period.

    Returns:
        Dict with 't' (seconds), 'e', 'n' (rad/s), 'a' (AU).
    """
    import rebound

    sa = rebound.Simulationarchive(bin_path)

    # Get orbital period from first snapshot
    sim0 = sa[0]
    orb0 = sim0.particles[planet_idx].orbit()
    p_orb_yr = orb0.P

    n_snapshots = len(sa)
    if n_orbits is not None:
        # Estimate how many snapshots cover n_orbits
        sim_last = sa[-1]
        total_time_yr = sim_last.t
        orbits_available = total_time_yr / p_orb_yr if p_orb_yr > 0 else 0
        if n_orbits > orbits_available:
            n_orbits = None  # use all

    # Sample the archive
    if n_orbits is not None:
        t_end = n_orbits * p_orb_yr
        n_samples = n_orbits * samples_per_orbit
    else:
        t_end = sa[-1].t
        n_samples = max(int(t_end / p_orb_yr * samples_per_orbit), 1000)

    times_yr = np.linspace(sa[0].t, t_end, n_samples)

    e_arr = np.zeros(n_samples)
    n_arr = np.zeros(n_samples)
    a_arr = np.zeros(n_samples)

    for i, t in enumerate(times_yr):
        try:
            sim = sa.getSimulation(t=t, mode="close")
        except Exception:
            # Past end of archive
            n_samples = i
            break
        orb = sim.particles[planet_idx].orbit()
        e_arr[i] = max(orb.e, 0.0)
        a_arr[i] = orb.a
        if orb.P > 0:
            n_arr[i] = 2.0 * np.pi / (orb.P * YEAR)
        else:
            n_arr[i] = n_arr[max(0, i - 1)]

    return {
        "t": times_yr[:n_samples] * YEAR,
        "e": e_arr[:n_samples],
        "n": n_arr[:n_samples],
        "a": a_arr[:n_samples],
    }


def main() -> None:
    """CLI entry point for standalone N-body extraction."""
    parser = argparse.ArgumentParser(
        description="Extract e(t), n(t) from N-body integration"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Short integration (10,000 orbits)")
    parser.add_argument("--planet", type=int, default=3,
                        help="Planet index (1=b, 2=c, 3=Bipolaris, 4=d)")
    parser.add_argument("--bin", type=str, default=None,
                        help="Path to REBOUND .bin archive (skip fresh sim)")
    parser.add_argument("--scenario", type=str, default=None,
                        help="Scenario name from shared/scenarios.py")
    args = parser.parse_args()

    n_orbits = 10_000 if args.quick else 10_000_000
    out_dir = output_dir_for(__file__, args.scenario)

    if args.bin:
        print(f"Loading archive: {args.bin}")
        result = load_rebound_archive(
            args.bin, args.planet, n_orbits=n_orbits,
        )
    else:
        overrides = None
        if args.scenario:
            from shared.scenarios import get_scenario
            overrides = get_scenario(args.scenario)

        print(f"Building Bipolaris system (planet {args.planet})...")
        sim = build_bipolaris_system(overrides)
        print(f"Integrating {n_orbits:,} orbits...")
        result = integrate_and_extract(sim, args.planet, n_orbits)

    # Save
    out_path = os.path.join(out_dir, f"nbody_planet{args.planet}.npz")
    np.savez(
        out_path,
        t=result["t"],
        e=result["e"],
        n=result["n"],
        a=result["a"],
    )
    print(f"Saved: {out_path}")
    print(f"  Duration: {result['t'][-1] / YEAR:.0f} years")
    print(f"  e range: [{result['e'].min():.6f}, {result['e'].max():.6f}]")
    print(f"  n range: [{result['n'].min():.4e}, {result['n'].max():.4e}] rad/s")


if __name__ == "__main__":
    main()
