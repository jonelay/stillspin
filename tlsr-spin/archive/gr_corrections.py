"""General Relativity corrections for N-body simulations.

GR apsidal precession is critical for close-in planets because it
can suppress Kozai-Lidov oscillations by maintaining fast periapse
precession. Without GR, inner planets may become unstable on shorter
timescales.

Uses reboundx for adding GR corrections to REBOUND simulations.

Key physics:
- GR precession rate: dω/dt = 3 G M_star n / (c² a (1 - e²))
- GR can quench Kozai when ω̇_GR >> ω̇_Kozai

References:
    Einstein 1915 (original derivation)
    Naoz+ 2013, ApJ 773, 187 (GR + Kozai interaction)
"""

import numpy as np

from shared.constants import (
    AU,
    BIPOLARIS_DISTANCE_AU,
    G,
    STAR_MASS,
    STAR_MASS_MSUN,
    YEAR,
    orbital_period_years,
)


# Speed of light in m/s
C_SI = 299792458.0


def gr_precession_rate(
    a_au: float,
    e: float,
    m_star_msun: float,
) -> float:
    """GR apsidal precession rate.

    The periapse precesses at:
        dω/dt = 3 G M / (c² a (1 - e²)) * n

    where n = 2π/P is the mean motion.

    Args:
        a_au: Semi-major axis (AU).
        e: Eccentricity.
        m_star_msun: Stellar mass (M_sun).

    Returns:
        Precession rate (rad/yr).
    """
    from shared.constants import M_SUN

    a_m = a_au * AU
    m_star = m_star_msun * M_SUN

    # Mean motion in rad/s
    n = np.sqrt(G * m_star / a_m**3)

    # GR precession rate in rad/s
    omega_dot = 3 * G * m_star * n / (C_SI**2 * a_m * (1 - e**2))

    # Convert to rad/yr
    return omega_dot * YEAR


def gr_precession_period(
    a_au: float,
    e: float,
    m_star_msun: float,
) -> float:
    """GR precession period (time for one full apsidal cycle).

    Args:
        a_au: Semi-major axis (AU).
        e: Eccentricity.
        m_star_msun: Stellar mass (M_sun).

    Returns:
        Precession period in years.
    """
    omega_dot = gr_precession_rate(a_au, e, m_star_msun)
    return 2 * np.pi / omega_dot


def gr_quenches_kozai(
    a_inner_au: float,
    e_inner: float,
    a_outer_au: float,
    m_star_msun: float,
    m_companion_msun: float,
) -> bool:
    """Check if GR precession quenches Kozai oscillations.

    GR quenches Kozai when ω̇_GR >> ω̇_Kozai (typically factor > 10).

    Args:
        a_inner_au: Inner planet semi-major axis (AU).
        e_inner: Inner planet eccentricity.
        a_outer_au: Companion semi-major axis (AU).
        m_star_msun: Primary mass (M_sun).
        m_companion_msun: Companion mass (M_sun).

    Returns:
        True if GR likely quenches Kozai oscillations.
    """
    from tlsr_spin.kozai import kozai_apsidal_precession_rate

    omega_gr = gr_precession_rate(a_inner_au, e_inner, m_star_msun)
    omega_kozai = kozai_apsidal_precession_rate(
        a_inner_au, a_outer_au, m_star_msun, m_companion_msun, e_inner
    )

    # GR quenches Kozai when it's dominant
    return abs(omega_gr) > 10 * abs(omega_kozai)


def add_gr_precession(sim: "rebound.Simulation") -> "reboundx.Extras":
    """Add GR apsidal precession to REBOUND simulation.

    Uses reboundx's gr_full effect which implements the full
    post-Newtonian equations (not just precession).

    Args:
        sim: REBOUND simulation to modify.

    Returns:
        reboundx.Extras object (for further configuration if needed).

    Raises:
        ImportError: If reboundx is not installed.
    """
    try:
        import reboundx
    except ImportError as e:
        raise ImportError(
            "reboundx required for GR corrections. Install with: pip install reboundx"
        ) from e

    rebx = reboundx.Extras(sim)

    # Use gr_full for accurate post-Newtonian corrections
    # (gr is simpler but less accurate for tight systems)
    gr = rebx.load_force("gr_full")
    rebx.add_force(gr)

    # Speed of light in simulation units
    # Simulation uses (yr, AU, Msun), so c needs conversion
    # c = 299792458 m/s = c_AU_per_yr AU/yr
    c_m_per_s = C_SI
    c_au_per_yr = c_m_per_s / AU * YEAR  # ~63241 AU/yr

    gr.params["c"] = c_au_per_yr

    return rebx


def build_bipolaris_system_with_gr(
    overrides: dict | None = None,
) -> "rebound.Simulation":
    """Build Bipolaris system with GR corrections enabled.

    Creates the standard 4-planet system and adds GR precession
    via reboundx.

    Args:
        overrides: Dict of constant overrides (from scenarios).

    Returns:
        Configured rebound.Simulation with GR enabled.
    """
    from tlsr_spin.nbody import build_bipolaris_system

    sim = build_bipolaris_system(overrides)
    add_gr_precession(sim)

    return sim


def print_gr_summary() -> None:
    """Print summary of GR effects for Bipolaris system."""
    from shared.constants import (
        BIPOLARIS_ECCENTRICITY,
        COMPANION_DISTANCE_AU,
        COMPANION_MASS_MSUN,
        PLANET_B_DISTANCE_AU,
        PLANET_B_ECCENTRICITY,
    )

    print("=== GR Precession Summary ===")

    planets = [
        ("b", PLANET_B_DISTANCE_AU, PLANET_B_ECCENTRICITY),
        ("Bipolaris", BIPOLARIS_DISTANCE_AU, BIPOLARIS_ECCENTRICITY),
    ]

    for name, a_au, e in planets:
        p_gr = gr_precession_period(a_au, e, STAR_MASS_MSUN)
        omega_gr = gr_precession_rate(a_au, e, STAR_MASS_MSUN)

        print(f"\n{name} (a={a_au:.3f} AU, e={e:.3f}):")
        print(f"  GR precession rate: {omega_gr:.4e} rad/yr")
        print(f"  GR precession period: {p_gr:.1f} yr")

        # Check Kozai quenching
        quenched = gr_quenches_kozai(
            a_au, e, COMPANION_DISTANCE_AU, STAR_MASS_MSUN, COMPANION_MASS_MSUN
        )
        print(f"  Kozai quenched by GR: {'YES' if quenched else 'NO'}")


if __name__ == "__main__":
    print_gr_summary()
