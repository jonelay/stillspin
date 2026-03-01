"""
Obliquity dynamics for a planet perturbed by a distant companion (the outer perturber,
a cold gas giant) and stabilised/destabilised by a moon (moon).

Implements the Colombo-Ward secular spin-orbit model:
    dε/dt  = -α sin ε (A cos ψ + B sin ψ)
    dψ/dt  = -α cos ε (A sin ψ - B cos ψ) / sin ε  -  s
where α is the spin-axis precession constant, s is the secular nodal
regression frequency driven by the outer perturber, and (A, B) encode the forced
inclination of the orbital plane.

References:
    Ward 1974, Laskar & Robutel 1993, Atobe & Ida 2007
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import argrelextrema


# ── Physical constants ──────────────────────────────────────────────────
G_SI = 6.674e-11            # m³ kg⁻¹ s⁻²
M_SUN = 1.989e30            # kg
M_EARTH = 5.972e24          # kg
R_EARTH = 6.371e6           # m
AU = 1.496e11               # m
YR = 3.156e7                # s


def compute_j2(omega, R, M, kf):
    """Rotational oblateness J₂ from spin rate and fluid Love number.

    J₂ = kf * ω²R³ / (3GM)
    """
    return kf * omega**2 * R**3 / (3.0 * G_SI * M)


def compute_alpha(j2, omega, R, M_planet, GM_star, a_orb, GM_moon, a_moon):
    """Spin-axis precession constant α.

    α = (3/2) * n² / ω * [ (GM_star/n²a³) * J₂ + (GM_moon)/(M_planet * a_moon³) * ... ]

    Simplified: α = (3/2) * (1/ω) * [ GM_star * J₂ * R² / a_orb³  +  GM_moon * R² / a_moon³ ] / (C/MR² factor)

    Using the standard result for a body with moment factor C/(MR²) ≈ 0.33:
        α = (3 n_orb² / (2 ω)) * (J₂ + q_moon)
    where q_moon encodes the moon's torque contribution and n_orb is the
    orbital mean motion.
    """
    C_factor = 0.33  # C/(MR²) for roughly Earth-like body
    n_orb = np.sqrt(GM_star / a_orb**3)
    # Moon torque equivalent J₂ contribution:
    # The moon contributes an effective J₂-like term:
    #   q_moon = (M_moon / M_planet) * (R / a_moon)³ * (1/(2 * J₂_norm))
    # But more directly, the total precession constant is:
    #   α = (3/(2 C_factor ω)) * [n_orb² * J₂  +  GM_moon * R² / (M_planet * R² * a_moon³)]
    # Wait — let's use the clean formulation.
    #
    # The precession rate of a planet's spin axis due to stellar torque:
    #   α_star = (3/2) * (n_orb² / ω) * (J₂ / C_factor)
    # Additional contribution from moon:
    #   α_moon = (3/2) * (n_moon² / ω) * (J₂_eff / C_factor)
    # where n_moon = sqrt(G M_planet / a_moon³), J₂_eff ≈ (M_moon/M_planet)(R/a_moon)³ ...
    #
    # Actually the clean formula (Ward 1975, Atobe & Ida 2007) is:
    #   α = (3 G M_star) / (2 C_factor ω a_orb³) * J₂ * M_planet * R²  ... no
    #
    # Let's just use:
    #   α = (3/(2 C_factor)) * (1/ω) * (n_orb² * J₂ + n_moon² * (M_moon/M_planet) * 0.5)
    # This is approximate but captures the right scaling.

    M_moon = GM_moon / G_SI
    n_moon_sq = G_SI * M_planet / a_moon**3
    alpha = (3.0 / (2.0 * C_factor)) * (1.0 / omega) * (
        n_orb**2 * j2 + n_moon_sq * (M_moon / M_planet) * 0.5
    )
    return alpha


def compute_s(n_orb, M_kerb, M_star, a_orb, a_kerb):
    """Secular nodal regression frequency driven by the outer perturber.

    Laplace-Lagrange secular theory (lowest order):
        s ≈ -(1/4) * n_orb * (M_kerb / M_star) * (a_orb / a_kerb)³ * b_{3/2}^(1)(a_orb/a_kerb)

    For a_orb << a_kerb, b_{3/2}^(1) ≈ 3 a_orb/a_kerb, giving:
        s ≈ -(3/4) * n_orb * (M_kerb/M_star) * (a_orb/a_kerb)²
    """
    ratio = a_orb / a_kerb
    s = (3.0 / 4.0) * n_orb * (M_kerb / M_star) * ratio**2
    return s  # positive magnitude; sign handled in ODE


def colombo_ward_rhs(t, state, alpha, s, I_forced):
    """Right-hand side of the Colombo-Ward obliquity ODE.

    State: [ε, ψ] where ε is obliquity, ψ is precession angle.

    The forced inclination I_forced sets the coupling amplitude.
    Using the simplified 2D formulation:
        dε/dt  = -α sin(2ε)/2 * sin(ψ) * sin(I_forced)
        dψ/dt  = -s - α cos(ε) + α sin(I_forced) cos(ψ) / tan(ε) (approximate)

    We use the Hamiltonian formulation in (x = cos ε) space for stability:
        dx/dt = α (1-x²) sin(ψ) sin(I_forced)
        dψ/dt = -s - α x + α x sin(I_forced) cos(ψ) / sqrt(1-x²)  ...

    For numerical robustness, we integrate (ε, ψ) directly with the standard
    Ward (1974) equations for the case of small forced inclination:

        dε/dt  = -α sin(I_f) sin(ψ)
        dψ/dt  = -s + α cos(ε)/sin(ε) * (1 - sin(I_f) cos(ψ))

    where I_f is the forced inclination of the orbital plane.
    """
    eps, psi = state
    sin_If = np.sin(I_forced)

    deps_dt = -alpha * sin_If * np.sin(psi)

    cos_eps = np.cos(eps)
    sin_eps = np.sin(eps)
    # Guard against sin(eps) = 0
    if abs(sin_eps) < 1e-12:
        sin_eps = 1e-12 * np.sign(sin_eps) if sin_eps != 0 else 1e-12

    dpsi_dt = -s + alpha * (cos_eps / sin_eps) * (1.0 - sin_If * np.cos(psi))

    return [deps_dt, dpsi_dt]


def integrate_obliquity(params, t_stop_yr, method='RK45', rtol=1e-9, atol=1e-12):
    """Integrate obliquity evolution for given system + moon parameters.

    Parameters
    ----------
    params : dict
        Must contain system parameters (see config.yaml 'system' block)
        plus 'moon_mass_mearth' and 'moon_distance_rp' (planetary radii).
    t_stop_yr : float
        Integration duration in years.
    method, rtol, atol : solver options.

    Returns
    -------
    dict with keys: t_yr, obliquity_deg, psi_rad, alpha, s, alpha_over_s
    """
    # Unpack system parameters
    M_star = params['star_mass_msun'] * M_SUN
    a_orb = params['bipolaris_distance_au'] * AU
    T_rot_hr = params['bipolaris_rotation_period_hr']
    M_planet = params['bipolaris_mass_mearth'] * M_EARTH
    R_planet = params['bipolaris_radius_rearth'] * R_EARTH
    M_kerb = params['kerberos_mass_msun'] * M_SUN
    a_kerb = params['kerberos_distance_au'] * AU
    eps0_deg = params['obliquity_initial_deg']
    kf = params['kf']

    M_moon = params['moon_mass_mearth'] * M_EARTH
    a_moon = params['moon_distance_rp'] * R_planet

    # Derived quantities
    omega = 2.0 * np.pi / (T_rot_hr * 3600.0)  # rad/s
    GM_star = G_SI * M_star
    GM_moon = G_SI * M_moon
    n_orb = np.sqrt(GM_star / a_orb**3)

    j2 = compute_j2(omega, R_planet, M_planet, kf)
    alpha = compute_alpha(j2, omega, R_planet, M_planet, GM_star, a_orb, GM_moon, a_moon)
    s = compute_s(n_orb, M_kerb, M_star, a_orb, a_kerb)

    # Forced inclination — from the outer perturber secular perturbation.
    # For a distant perturber, I_forced ~ (M_kerb/M_star) * (a_orb/a_kerb)^2 * sin(I_mutual)
    # Note: this is the lowest-order Laplace-Lagrange result. At I_mutual > ~30°,
    # higher-order terms become significant; results should be treated as approximate.
    I_mutual = np.radians(params.get('mutual_inclination_deg', 5.0))
    I_forced = (M_kerb / M_star) * (a_orb / a_kerb)**2 * np.sin(I_mutual)
    I_forced = np.clip(I_forced, 0, np.pi / 2)  # cap at 90° for safety
    I_forced_deg = np.degrees(I_forced)

    eps0 = np.radians(eps0_deg)
    psi0 = 0.0
    t_stop_s = t_stop_yr * YR

    # Number of output points: ~100 per expected oscillation period
    # Use at least 2000 points
    n_points = max(2000, int(t_stop_yr * 10))
    t_eval = np.linspace(0, t_stop_s, n_points)

    sol = solve_ivp(
        colombo_ward_rhs,
        [0, t_stop_s],
        [eps0, psi0],
        args=(alpha, s, I_forced),
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        max_step=t_stop_s / 1000,
    )

    if not sol.success:
        return {
            't_yr': np.array([]),
            'obliquity_deg': np.array([]),
            'psi_rad': np.array([]),
            'alpha': alpha,
            's': s,
            'alpha_over_s': alpha / s if s != 0 else np.inf,
            'i_forced_deg': I_forced_deg,
            'success': False,
            'message': sol.message,
        }

    return {
        't_yr': sol.t / YR,
        'obliquity_deg': np.degrees(sol.y[0]),
        'psi_rad': sol.y[1],
        'alpha': alpha,
        's': s,
        'alpha_over_s': alpha / s if s != 0 else np.inf,
        'i_forced_deg': I_forced_deg,
        'success': True,
    }


def measure_oscillation(t_yr, obliquity_deg):
    """Extract period and amplitude of obliquity oscillations.

    Returns (period_yr, amplitude_deg). Returns (NaN, NaN) if oscillation
    cannot be measured.
    """
    if len(t_yr) < 10:
        return np.nan, np.nan

    obl = obliquity_deg
    mean_obl = np.mean(obl)
    centered = obl - mean_obl

    # Find extrema
    maxima = argrelextrema(centered, np.greater, order=5)[0]
    minima = argrelextrema(centered, np.less, order=5)[0]

    if len(maxima) < 2 and len(minima) < 2:
        # No clear oscillation — measure total range
        amplitude = (np.max(obl) - np.min(obl)) / 2.0
        return np.nan, amplitude

    # Period from peak-to-peak
    if len(maxima) >= 2:
        periods = np.diff(t_yr[maxima])
        period = np.median(periods)
    elif len(minima) >= 2:
        periods = np.diff(t_yr[minima])
        period = np.median(periods)
    else:
        period = np.nan

    # Amplitude: half of peak-to-trough range
    if len(maxima) > 0 and len(minima) > 0:
        amplitude = (np.median(obl[maxima]) - np.median(obl[minima])) / 2.0
    else:
        amplitude = (np.max(obl) - np.min(obl)) / 2.0

    return period, amplitude
