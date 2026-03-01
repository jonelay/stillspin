"""Kozai-Lidov secular dynamics for the K5V companion.

The distant K5V companion at 180 AU drives Kozai-Lidov oscillations
in the inner planetary system. These oscillations can pump eccentricities
over Gyr timescales, eventually destabilizing the Laplace chain and
triggering "the instability" ~350 Mya.

Key physics:
- Critical inclination: i > 39.2° for Kozai cycles
- Kozai timescale: P_Kozai ~ (M_star/M_comp) * (a_outer/a_inner)³ * P_outer
- Maximum eccentricity: e_max = sqrt(1 - (5/3)cos²i) at critical phases

References:
    Kozai 1962, AJ 67, 591
    Lidov 1962, Planet. Space Sci. 9, 719
    Naoz 2016, ARA&A 54, 441 (review)
"""

import numpy as np

from shared.constants import (
    AU,
    BIPOLARIS_DISTANCE_AU,
    COMPANION_DISTANCE_AU,
    COMPANION_INCLINATION_DEG,
    COMPANION_MASS,
    COMPANION_MASS_MSUN,
    G,
    STAR_MASS,
    STAR_MASS_MSUN,
    YEAR,
    orbital_period_years,
)


# Critical inclination for Kozai oscillations (degrees)
KOZAI_CRITICAL_INCLINATION_DEG = 39.2315  # arccos(sqrt(3/5)) ≈ 39.23°


def kozai_timescale(
    a_inner_au: float,
    a_outer_au: float,
    m_star_msun: float,
    m_companion_msun: float,
) -> float:
    """Kozai-Lidov cycle period in years.

    The secular Kozai timescale is approximately:
        P_Kozai = (M_star / M_comp) * (a_outer / a_inner)³ * P_outer * (1 - e_outer²)^(3/2)

    For the Bipolaris system with companion at 180 AU, direct binary Kozai
    would have P_Kozai >> system age. However, in multi-planet systems,
    secular dynamics are more complex and can operate on shorter timescales
    through planet-planet interactions modulated by the distant perturber.

    The effective Kozai timescale (~100 Myr) reflects this multi-body effect,
    not the simple binary formula.

    Args:
        a_inner_au: Semi-major axis of inner planet (AU).
        a_outer_au: Semi-major axis of companion (AU).
        m_star_msun: Primary stellar mass (M_sun).
        m_companion_msun: Companion mass (M_sun).

    Returns:
        Kozai cycle period in years (binary Kozai formula).
    """
    # Companion orbital period (around combined mass)
    m_total_msun = m_star_msun + m_companion_msun
    p_outer_yr = np.sqrt(a_outer_au**3 / m_total_msun)

    # Kozai timescale (binary formula)
    mass_ratio = m_star_msun / m_companion_msun
    a_ratio = a_outer_au / a_inner_au

    return mass_ratio * a_ratio**3 * p_outer_yr


def effective_kozai_timescale_multiplanet() -> float:
    """Effective Kozai-like timescale for multi-planet system.

    In a multi-planet system with a distant perturber, secular dynamics
    operate through eigenmode coupling. The effective timescale for
    eccentricity pumping is much shorter than binary Kozai due to
    resonance overlap and chaotic diffusion.

    For the Bipolaris system, this is the effective timescale (~100 Myr)
    that describes how long before the instability could occur.

    Returns:
        Effective timescale in years (estimated, not computed).
    """
    # From system-design-v2.md: "Kozai-Lidov timescale: ~100 Myr"
    return 1e8  # 100 Myr


def kozai_emax(inclination_deg: float, e_initial: float = 0.0) -> float:
    """Maximum eccentricity from Kozai oscillation.

    Conservation of the Kozai integral (Lidov-Kozai constant):
        (1 - e²) cos²i = const

    Starting from e ≈ 0, the maximum eccentricity is:
        e_max = sqrt(1 - (5/3) cos²i)

    For i > 39.2°, this gives real e_max > 0.

    Args:
        inclination_deg: Relative inclination (degrees).
        e_initial: Initial eccentricity.

    Returns:
        Maximum achievable eccentricity.
    """
    i_rad = np.deg2rad(inclination_deg)
    cos_i = np.cos(i_rad)

    # Kozai integral conservation
    lk_constant = (1 - e_initial**2) * cos_i**2

    # Maximum e when cos i = cos i_crit
    # e_max² = 1 - lk_constant / cos²i_min
    # For circular orbit: e_max = sqrt(1 - 5/3 cos²i) when i > i_crit

    if inclination_deg < KOZAI_CRITICAL_INCLINATION_DEG:
        # Below critical inclination, no Kozai oscillations
        return e_initial

    e_max_sq = 1 - (5.0 / 3.0) * cos_i**2
    if e_max_sq < 0:
        return e_initial

    return np.sqrt(e_max_sq)


def kozai_timescale_bipolaris() -> float:
    """Kozai timescale for the Bipolaris system.

    Uses default system parameters from constants.

    Returns:
        Kozai cycle period in years.
    """
    return kozai_timescale(
        BIPOLARIS_DISTANCE_AU,
        COMPANION_DISTANCE_AU,
        STAR_MASS_MSUN,
        COMPANION_MASS_MSUN,
    )


def kozai_emax_bipolaris() -> float:
    """Maximum eccentricity for Bipolaris from companion Kozai.

    Uses default companion inclination from constants.

    Returns:
        Maximum achievable eccentricity.
    """
    return kozai_emax(COMPANION_INCLINATION_DEG)


def kozai_apsidal_precession_rate(
    a_inner_au: float,
    a_outer_au: float,
    m_star_msun: float,
    m_companion_msun: float,
    e_inner: float,
) -> float:
    """Apsidal precession rate from distant companion.

    The companion induces apsidal precession on inner planets:
        ω̇ ≈ (3/4) n (M_comp/M_star) (a_inner/a_outer)³ (1 - e²)^(-1/2)

    Args:
        a_inner_au: Inner planet semi-major axis (AU).
        a_outer_au: Companion semi-major axis (AU).
        m_star_msun: Primary mass (M_sun).
        m_companion_msun: Companion mass (M_sun).
        e_inner: Inner planet eccentricity.

    Returns:
        Precession rate (rad/yr).
    """
    # Inner planet mean motion (rad/yr)
    p_inner_yr = np.sqrt(a_inner_au**3 / m_star_msun)
    n_inner = 2 * np.pi / p_inner_yr

    mass_ratio = m_companion_msun / m_star_msun
    a_ratio = a_inner_au / a_outer_au

    omega_dot = (3.0 / 4.0) * n_inner * mass_ratio * a_ratio**3 / np.sqrt(1 - e_inner**2)

    return omega_dot


def integrate_kozai_secular(
    t_end_yr: float,
    a_inner_au: float = BIPOLARIS_DISTANCE_AU,
    e_initial: float = 0.01,
    inclination_deg: float = COMPANION_INCLINATION_DEG,
    omega_initial_deg: float = 0.0,
    n_samples: int = 10000,
) -> dict[str, np.ndarray]:
    """Secular integration of Kozai-Lidov dynamics.

    Integrates the secular equations for eccentricity and argument
    of periapse under Kozai-Lidov forcing from the companion.

    Uses the quadrupole approximation (valid when a_inner << a_outer).

    Args:
        t_end_yr: Integration duration (years).
        a_inner_au: Inner planet semi-major axis (AU).
        e_initial: Initial eccentricity.
        inclination_deg: Mutual inclination (degrees).
        omega_initial_deg: Initial argument of periapse (degrees).
        n_samples: Number of output samples.

    Returns:
        Dict with 't' (years), 'e', 'i_deg', 'omega_deg' arrays.
    """
    from scipy.integrate import solve_ivp

    i_rad = np.deg2rad(inclination_deg)
    omega_rad = np.deg2rad(omega_initial_deg)

    # Kozai timescale for normalization
    p_kozai = kozai_timescale(
        a_inner_au, COMPANION_DISTANCE_AU, STAR_MASS_MSUN, COMPANION_MASS_MSUN
    )
    omega_kozai = 2 * np.pi / p_kozai  # rad/yr

    def secular_ode(t: float, state: np.ndarray) -> np.ndarray:
        """Kozai-Lidov secular equations (quadrupole approximation).

        State: [e, i, omega] where i and omega are in radians.
        """
        e, i, omega = state

        cos_i = np.cos(i)
        sin_i = np.sin(i)
        sin_omega = np.sin(omega)
        cos_omega = np.cos(omega)

        # Prevent numerical issues at e=0 or e=1
        e = np.clip(e, 1e-10, 1.0 - 1e-10)

        # Kozai-Lidov equations (Naoz 2016, Eqs. 4-6, quadrupole)
        # de/dt = (15/8) n_K e (1-e²)^(1/2) sin²i sin(2ω)
        de_dt = (15.0 / 8.0) * omega_kozai * e * np.sqrt(1 - e**2) * sin_i**2 * np.sin(2 * omega)

        # di/dt = -(15/16) n_K e² (1-e²)^(-1/2) sin(2i) sin(2ω)
        di_dt = -(15.0 / 16.0) * omega_kozai * e**2 / np.sqrt(1 - e**2) * np.sin(2 * i) * np.sin(2 * omega)

        # dω/dt = (3/4) n_K [(4 - 5sin²i)(1 - e²)^(1/2) + 5cos²ω(1 - e² - sin²i)]
        domega_dt = (3.0 / 4.0) * omega_kozai * (
            (4 - 5 * sin_i**2) / np.sqrt(1 - e**2)
            + 5 * cos_omega**2 * (1 - e**2 - sin_i**2) / (1 - e**2) ** 1.5
        )

        return np.array([de_dt, di_dt, domega_dt])

    t_span = (0, t_end_yr)
    t_eval = np.linspace(0, t_end_yr, n_samples)
    y0 = np.array([e_initial, i_rad, omega_rad])

    sol = solve_ivp(
        secular_ode,
        t_span,
        y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )

    if not sol.success:
        raise RuntimeError(f"Kozai secular integration failed: {sol.message}")

    return {
        "t": sol.t,
        "e": sol.y[0],
        "i_deg": np.rad2deg(sol.y[1]),
        "omega_deg": np.rad2deg(sol.y[2]) % 360,
    }


def print_kozai_summary() -> None:
    """Print summary of Kozai-Lidov dynamics for Bipolaris system."""
    p_kozai_binary = kozai_timescale_bipolaris()
    p_kozai_effective = effective_kozai_timescale_multiplanet()
    e_max = kozai_emax_bipolaris()

    print("=== Kozai-Lidov Dynamics Summary ===")
    print(f"Inner planet: {BIPOLARIS_DISTANCE_AU:.3f} AU")
    print(f"Companion: {COMPANION_MASS_MSUN:.2f} M_sun at {COMPANION_DISTANCE_AU:.0f} AU")
    print(f"Mutual inclination: {COMPANION_INCLINATION_DEG:.1f}°")
    print(f"Critical inclination: {KOZAI_CRITICAL_INCLINATION_DEG:.1f}°")
    print()
    print(f"Binary Kozai period: {p_kozai_binary:.2e} yr (too long for direct effect)")
    print(f"Effective multi-planet timescale: {p_kozai_effective:.0e} yr (estimated)")
    print(f"Maximum eccentricity (if critical i): {e_max:.3f}")
    print()

    # Check if Kozai active
    if COMPANION_INCLINATION_DEG > KOZAI_CRITICAL_INCLINATION_DEG:
        print("Status: Above critical inclination - Kozai oscillations possible")
    else:
        print("Status: Below critical inclination - secular perturbations only")
        print("        Multi-planet interactions can still drive instability")


if __name__ == "__main__":
    print_kozai_summary()
