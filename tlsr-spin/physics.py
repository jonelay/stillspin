"""
Spin-orbit dynamics for tidally locked planets with secular resonance.

Implements the Goldreich & Peale (1966) spin equation as used in
Shakespeare & Steffen (2023) DayNite pipeline. The key equation is:

    γ̈ + (1/2) ω_s² sin(2γ) + ṅ + ε γ̇ = 0

where γ is the angle between the substellar point and the planet's
long axis (γ = φ - M, spin angle minus mean anomaly), ω_s² is the
restoring torque from triaxiality, ṅ is the time derivative of mean
motion (from secular perturbations), and ε is the tidal dissipation
coefficient.

References:
    Goldreich & Peale 1966, AJ 71, 425
    Shakespeare & Steffen 2023, ApJ 959, 170
    MacDonald & Dawson 2018, AJ 156, 228
"""

import numpy as np
from scipy.interpolate import CubicSpline


def eccentricity_function(e: float) -> float:
    """Eccentricity correction to the triaxial restoring torque.

    H(e) = 1 - 5e²/2 + 13e⁴/16  (Goldreich & Peale 1966, Eq. 2)

    This modifies the strength of the synchronous spin-orbit resonance.
    H(0) = 1 (circular orbit), decreases with eccentricity.

    Args:
        e: Orbital eccentricity.

    Returns:
        H(e) dimensionless correction factor.
    """
    return 1.0 - 2.5 * e**2 + (13.0 / 16.0) * e**4


def omega_s_squared(
    n: float,
    triaxiality: float,
    e: float,
) -> float:
    """Restoring torque frequency squared for synchronous resonance.

    ω_s² = 3 n² (B-A)/C |H(e)|

    where (B-A)/C is the triaxiality parameter and n is the mean motion.

    Args:
        n: Orbital mean motion (rad/s).
        triaxiality: (B-A)/C, dimensionless moment-of-inertia difference.
        e: Orbital eccentricity.

    Returns:
        ω_s² in rad²/s².
    """
    h_e = eccentricity_function(e)
    return 3.0 * n**2 * triaxiality * abs(h_e)


def tidal_epsilon(
    m_star: float,
    m_planet: float,
    r_planet: float,
    a: float,
    tidal_q: float,
    omega_spin: float,
) -> float:
    """Tidal dissipation coefficient (constant time-lag model).

    ε = (15/2) (M★/m) (R/a)⁶ M★ R² σ

    where σ = G / (2 Q Ω R⁵) is the tidal dissipation function
    (Shakespeare & Steffen 2023, Eq. 7; Goldreich & Soter 1966).

    Substituting σ:
        ε = (15/4) G M★² R³ / (Q Ω m a⁶)

    Note: Ω here is the spin rate, typically ≈ n for near-synchronous.

    Args:
        m_star: Stellar mass (kg).
        m_planet: Planet mass (kg).
        r_planet: Planet radius (m).
        a: Semi-major axis (m).
        tidal_q: Tidal quality factor (dimensionless).
        omega_spin: Spin angular velocity (rad/s). Use n for synchronous.

    Returns:
        Dissipation coefficient ε (1/s).
    """
    from shared.constants import G as G_SI

    sigma = G_SI / (2.0 * tidal_q * omega_spin * r_planet**5)
    eps = (15.0 / 2.0) * (m_star / m_planet) * (r_planet / a) ** 6 * m_star * r_planet**2 * sigma
    return eps


def spin_ode(
    t: float,
    state: np.ndarray,
    e_spline: CubicSpline,
    n_spline: CubicSpline,
    triaxiality: float,
    eps: float,
) -> list[float]:
    """Right-hand side of the spin-orbit ODE.

    State vector: [γ, γ̇] where γ = φ - M (spin angle minus mean anomaly).

    The equation of motion (Goldreich & Peale 1966, Shakespeare & Steffen 2023):
        γ̈ = -(1/2) ω_s² sin(2γ) - ṅ - ε γ̇

    where ω_s² = 3 n² (B-A)/C |H(e)|, and ṅ is obtained from the
    derivative of the mean-motion spline.

    Args:
        t: Time (seconds).
        state: [γ, γ̇] in radians and rad/s.
        e_spline: Cubic spline interpolant for e(t).
        n_spline: Cubic spline interpolant for n(t).
        triaxiality: (B-A)/C dimensionless.
        eps: Tidal dissipation coefficient (1/s).

    Returns:
        [γ̇, γ̈] derivatives.
    """
    gamma, gamma_dot = state

    e = float(e_spline(t))
    n = float(n_spline(t))
    n_dot = float(n_spline(t, 1))  # first derivative of spline

    w_s2 = omega_s_squared(n, triaxiality, e)

    gamma_ddot = -0.5 * w_s2 * np.sin(2.0 * gamma) - n_dot - eps * gamma_dot

    return [gamma_dot, gamma_ddot]
