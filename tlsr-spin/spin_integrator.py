"""
Spin integration driver for TLSR dynamics.

Takes time series of eccentricity e(t) and mean motion n(t) from N-body
integration, constructs cubic spline interpolants, and integrates the
Goldreich & Peale spin ODE using scipy's solve_ivp.

Replaces the custom RK4 integrator with overlapping chunks from the
DayNite calcSpinX.py script.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

from .physics import spin_ode, tidal_epsilon


def integrate_spin(
    times: np.ndarray,
    e_t: np.ndarray,
    n_t: np.ndarray,
    m_star: float,
    m_planet: float,
    r_planet: float,
    a_mean: float,
    tidal_q: float,
    triaxiality: float,
    gamma_0: float = 0.0,
    gamma_dot_0: float = 0.0,
    n_output: int | None = None,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Integrate the spin-orbit ODE over the provided orbital history.

    Args:
        times: Time array (seconds) from N-body integration.
        e_t: Eccentricity at each time.
        n_t: Mean motion (rad/s) at each time.
        m_star: Stellar mass (kg).
        m_planet: Planet mass (kg).
        r_planet: Planet radius (m).
        a_mean: Mean semi-major axis (m), for tidal epsilon calculation.
        tidal_q: Tidal quality factor.
        triaxiality: (B-A)/C moment-of-inertia asymmetry.
        gamma_0: Initial spin-orbit angle γ (rad). Default 0 (synchronous).
        gamma_dot_0: Initial γ̇ (rad/s). Default 0.
        n_output: Number of output points. Default: same as input.
        rtol: Relative tolerance for solve_ivp.
        atol: Absolute tolerance for solve_ivp.

    Returns:
        Dict with keys: 't' (s), 'gamma' (rad), 'gamma_dot' (rad/s),
        'e', 'n' (interpolated at output times).
    """
    # Build cubic spline interpolants
    e_spline = CubicSpline(times, e_t)
    n_spline = CubicSpline(times, n_t)

    # Compute tidal dissipation coefficient
    # Use median n as representative spin rate (near-synchronous)
    omega_spin = float(np.median(n_t))
    eps = tidal_epsilon(m_star, m_planet, r_planet, a_mean, tidal_q, omega_spin)

    # Output time grid
    if n_output is None:
        n_output = len(times)
    t_eval = np.linspace(times[0], times[-1], n_output)

    # Integrate using dense output (avoids evaluating at many points)
    # Then sample at desired output times
    sol = solve_ivp(
        spin_ode,
        [times[0], times[-1]],
        [gamma_0, gamma_dot_0],
        args=(e_spline, n_spline, triaxiality, eps),
        method="RK45",
        dense_output=True,
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(f"Spin integration failed: {sol.message}")

    # Sample at desired output times using interpolating function
    if n_output is None or n_output == len(times):
        # If no downsampling requested, use all solver evaluation points
        t_sample = sol.t
        gamma_sample = sol.y[0]
        gamma_dot_sample = sol.y[1]
    else:
        # Use dense output to interpolate at desired times
        t_sample = np.linspace(times[0], times[-1], n_output)
        sol_interp = sol.sol(t_sample)
        gamma_sample = sol_interp[0]
        gamma_dot_sample = sol_interp[1]

    return {
        "t": t_sample,
        "gamma": gamma_sample,
        "gamma_dot": gamma_dot_sample,
        "e": np.array([float(e_spline(ti)) for ti in t_sample]),
        "n": np.array([float(n_spline(ti)) for ti in t_sample]),
        "tidal_epsilon": eps,
    }
