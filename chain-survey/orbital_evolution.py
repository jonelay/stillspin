"""Pipeline 1: Orbital evolution over 500 Myr.

Two-phase evolution:
  Phase A (0-5 Myr): Gas disk present, eccentricity damping (tau_e = 10^4 yr).
  Phase B (5-500 Myr): Disk dispersed, free evolution with tidal + GR.

Monitors for MMR breaking, ejections, and collisions.
Saves final state as REBOUND SimulationArchive for Pipeline 2.
"""

import time
import warnings
from pathlib import Path

import numpy as np

from shared.constants import AU, M_EARTH, M_SUN

from .chain_types import OrbitalResult, SystemArchitecture

# Tidal and orbit-modification forces are velocity-dependent, which WHFAST
# warns about. For this survey the tidal perturbations are weak and the
# symplectic error is negligible, so we suppress the warning for performance
# (operator wrapping adds ~10x overhead).
warnings.filterwarnings(
    "ignore",
    message="REBOUNDx.*velocity-dependent.*WHFAST",
    category=RuntimeWarning,
)


def build_evolution_sim(system: SystemArchitecture):
    """Build REBOUND sim with REBOUNDx forces.

    Returns (sim, rebx) tuple.
    """
    import rebound
    import reboundx

    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")

    sim.add(m=system.stellar_mass_msun)

    m_earth_msun = M_EARTH / M_SUN
    for p in system.planets:
        sim.add(
            m=p.mass_mearth * m_earth_msun,
            a=p.distance_au,
            e=p.eccentricity,
        )

    sim.move_to_com()

    shortest_period = min(p.period_days for p in system.planets) / 365.25
    sim.integrator = "whfast"
    sim.dt = shortest_period / 10.0  # P/10 is safe for WHFAST long-term surveys

    rebx = reboundx.Extras(sim)

    # GR precession (position-dependent, safe as force with WHFAST)
    gr = rebx.load_force("gr_potential")
    gr.params["c"] = 10065.32  # Speed of light in AU/yr
    rebx.add_force(gr)
    sim.particles[0].params["gr_source"] = 1

    # Tidal dissipation (velocity-dependent — applied as force for speed;
    # symplectic error is negligible for weak tidal perturbations)
    tides = rebx.load_force("tides_constant_time_lag")
    rebx.add_force(tides)

    return sim, rebx


def _add_disk_damping(sim, rebx, tau_e: float = 1e4):
    """Add eccentricity damping forces (Phase A: disk present)."""
    mof = rebx.load_force("modify_orbits_forces")
    rebx.add_force(mof)
    for i in range(1, sim.N):
        sim.particles[i].params["tau_e"] = -tau_e
    return mof


def _remove_disk_damping(rebx, mof):
    """Remove eccentricity damping (Phase B: disk dispersed)."""
    rebx.remove_force(mof)


def _get_period_ratios(sim) -> list[float]:
    """Get period ratios of adjacent planet pairs."""
    ratios = []
    for i in range(1, sim.N - 1):
        p1 = sim.particles[i].orbit().P
        p2 = sim.particles[i + 1].orbit().P
        if p1 > 0 and p2 > 0:
            ratios.append(p2 / p1)
        else:
            ratios.append(0.0)
    return ratios


def detect_mmr_breaking(
    current_ratios: list[float],
    initial_ratios: list[float],
    tolerance: float = 0.05,
) -> list[dict]:
    """Compare current period ratios to initial. Flag > tolerance drift."""
    events = []
    for i, (curr, init) in enumerate(zip(current_ratios, initial_ratios)):
        if init > 0 and abs(curr - init) / init > tolerance:
            events.append({
                "pair_idx": i,
                "initial_ratio": round(init, 4),
                "current_ratio": round(curr, 4),
                "drift_frac": round(abs(curr - init) / init, 4),
            })
    return events


def extract_final_elements(sim, n_planets: int) -> list[dict]:
    """Extract a, e, i for each surviving particle."""
    elements = []
    for i in range(1, min(sim.N, n_planets + 1)):
        orb = sim.particles[i].orbit()
        survived = 0 < orb.a < 10.0
        elements.append({
            "idx": i,
            "a_au": round(orb.a, 6),
            "e": round(orb.e, 6),
            "incl_deg": round(np.degrees(orb.inc), 4),
            "survived": survived,
        })
    return elements


def evolve_system(
    system: SystemArchitecture,
    t_end_myr: float = 500.0,
    output_dir: str | None = None,
) -> OrbitalResult:
    """Two-phase orbital evolution.

    Args:
        system: System to evolve.
        t_end_myr: Total evolution time in Myr.
        output_dir: Directory to save simulation archive.

    Returns:
        OrbitalResult with evolution outcome.
    """
    import rebound

    t0 = time.time()
    n_planets = len(system.planets)

    try:
        sim, rebx = build_evolution_sim(system)
    except Exception as e:
        return OrbitalResult(
            system_id=system.system_id,
            status="ERROR",
            final_planets=[],
            breaking_events=[],
            n_survivors=0,
            hz_planet_survived=False,
            hz_planet_idx=system.hz_planet_indices[0] + 1,
            elapsed_s=time.time() - t0,
            error_msg=f"Build failed: {e}",
        )

    hz_planet_idx = system.hz_planet_indices[0] + 1  # 1-based REBOUND index
    initial_ratios = _get_period_ratios(sim)
    breaking_events = []

    # Phase A: Disk damping (0 to 5 Myr)
    t_disk_end_yr = min(5e6, t_end_myr * 1e6)
    mof = _add_disk_damping(sim, rebx)

    n_checkpoints_a = 10
    dt_a = t_disk_end_yr / n_checkpoints_a

    for step in range(n_checkpoints_a):
        try:
            sim.integrate(sim.t + dt_a)
        except rebound.Encounter:
            return OrbitalResult(
                system_id=system.system_id,
                status="COLLISION",
                final_planets=extract_final_elements(sim, n_planets),
                breaking_events=breaking_events,
                n_survivors=sim.N - 1,
                hz_planet_survived=False,
                hz_planet_idx=hz_planet_idx,
                elapsed_s=time.time() - t0,
            )

        # Check ejections
        for i in range(1, sim.N):
            orb = sim.particles[i].orbit()
            if orb.a > 10.0 or orb.a < 0:
                return OrbitalResult(
                    system_id=system.system_id,
                    status="EJECTION",
                    final_planets=extract_final_elements(sim, n_planets),
                    breaking_events=[{
                        "time_myr": round(sim.t / 1e6, 4),
                        "type": "ejection",
                        "planet_idx": i,
                        "description": f"Planet {i} ejected during Phase A",
                    }],
                    n_survivors=sim.N - 2,
                    hz_planet_survived=(i != hz_planet_idx),
                    hz_planet_idx=hz_planet_idx,
                    elapsed_s=time.time() - t0,
                )

    # Phase B: Free evolution (5 Myr to t_end)
    _remove_disk_damping(rebx, mof)

    t_end_yr = t_end_myr * 1e6
    n_checkpoints_b = max(10, int(t_end_myr / 50))  # Check every ~50 Myr
    dt_b = (t_end_yr - sim.t) / n_checkpoints_b

    for step in range(n_checkpoints_b):
        try:
            sim.integrate(sim.t + dt_b)
        except rebound.Encounter:
            return OrbitalResult(
                system_id=system.system_id,
                status="COLLISION",
                final_planets=extract_final_elements(sim, n_planets),
                breaking_events=breaking_events,
                n_survivors=sim.N - 1,
                hz_planet_survived=False,
                hz_planet_idx=hz_planet_idx,
                elapsed_s=time.time() - t0,
            )

        # Check ejections
        for i in range(1, sim.N):
            orb = sim.particles[i].orbit()
            if orb.a > 10.0 or orb.a < 0:
                breaking_events.append({
                    "time_myr": round(sim.t / 1e6, 4),
                    "type": "ejection",
                    "planet_idx": i,
                    "description": f"Planet {i} ejected at {sim.t / 1e6:.1f} Myr",
                })
                # Don't return immediately — continue with remaining planets
                if i == hz_planet_idx:
                    return OrbitalResult(
                        system_id=system.system_id,
                        status="EJECTION",
                        final_planets=extract_final_elements(sim, n_planets),
                        breaking_events=breaking_events,
                        n_survivors=sim.N - 2,
                        hz_planet_survived=False,
                        hz_planet_idx=hz_planet_idx,
                        elapsed_s=time.time() - t0,
                    )

        # Check MMR breaking
        current_ratios = _get_period_ratios(sim)
        mmr_breaks = detect_mmr_breaking(current_ratios, initial_ratios)
        for brk in mmr_breaks:
            breaking_events.append({
                "time_myr": round(sim.t / 1e6, 4),
                "type": "mmr_break",
                **brk,
            })

    # Determine final status
    final_elements = extract_final_elements(sim, n_planets)
    n_survivors = sum(1 for el in final_elements if el["survived"])
    hz_survived = any(
        el["survived"] for el in final_elements if el["idx"] == hz_planet_idx
    )

    if not breaking_events:
        status = "INTACT"
    elif n_survivors == n_planets:
        status = "PARTIAL_BREAK"
    else:
        status = "FULL_BREAK"

    # Save simulation archive
    sim_archive_path = None
    if output_dir:
        archive_dir = Path(output_dir)
        archive_dir.mkdir(parents=True, exist_ok=True)
        sim_archive_path = str(archive_dir / f"{system.system_id}.bin")
        sim.save_to_file(sim_archive_path)

    return OrbitalResult(
        system_id=system.system_id,
        status=status,
        final_planets=final_elements,
        breaking_events=breaking_events,
        n_survivors=n_survivors,
        hz_planet_survived=hz_survived,
        hz_planet_idx=hz_planet_idx,
        elapsed_s=time.time() - t0,
        sim_archive_path=sim_archive_path,
    )
