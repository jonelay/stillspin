"""
Single source of truth for Bipolaris system parameters.

System Design v2: Broken Chain Architecture

M5.5V red dwarf system with a distant K5V companion.
Four inner rocky planets remain from an original five-planet Laplace chain.
Planet e was ejected ~500 Mya when Kozai-Lidov cycles from the companion
pumped its eccentricity past the stability threshold.

Bipolaris orbits at 0.0745 AU in the middle of the HZ. For 7+ Gyr it was
tidally locked with a fixed substellar point. After a late instability, it entered
a flip-flop regime where the substellar point irregularly shifts between
two stable orientations (TL_ZERO and TL_PI) on decadal timescales, with
brief ~1 year transitional periods between locks.

A small (~300 km) debris moon orbits at 5 R_E with an 18-hour period.

All values in SI unless noted. Orbital distances in AU where marked.
"""

import numpy as np

# --- Fundamental constants ---
M_SUN = 1.989e30       # kg
M_EARTH = 5.972e24     # kg
M_JUPITER = 1.898e27   # kg
R_EARTH = 6.371e6      # m
L_SUN = 3.828e26       # W
AU = 1.496e11          # m
G = 6.674e-11          # m^3 kg^-1 s^-2
YEAR = 365.25 * 86400  # seconds
DAY = 86400            # seconds

# --- Primary star (M5.5V) ---
STAR_MASS = 0.15 * M_SUN            # kg
STAR_MASS_MSUN = 0.15
STAR_LUMINOSITY = 0.0022 * L_SUN    # W — typical M5.5V
STAR_LUMINOSITY_LSUN = 0.0022
STAR_TEFF = 2900                    # K — effective temperature
STAR_SPECTRAL_TYPE = "M5.5V"
SYSTEM_AGE_GYR = 7.5                # Gyr

# --- Distant companion (K5V, Kozai-Lidov driver) ---
COMPANION_MASS = 0.55 * M_SUN       # kg
COMPANION_MASS_MSUN = 0.55
COMPANION_DISTANCE_AU = 180.0       # AU from primary
COMPANION_INCLINATION_DEG = 35.0    # relative to inner system

# --- Planet b (hot rocky) ---
PLANET_B_DISTANCE_AU = 0.018
PLANET_B_MASS = 0.8 * M_EARTH
PLANET_B_RADIUS = 0.93 * R_EARTH
PLANET_B_ECCENTRICITY = 0.03

# --- Planet c (inner HZ edge) ---
PLANET_C_DISTANCE_AU = 0.032
PLANET_C_MASS = 1.4 * M_EARTH
PLANET_C_RADIUS = 1.10 * R_EARTH
PLANET_C_ECCENTRICITY = 0.04

# --- Bipolaris (middle of HZ) ---
BIPOLARIS_MASS = 1.0 * M_EARTH
BIPOLARIS_RADIUS = 1.06 * R_EARTH
BIPOLARIS_DISTANCE_AU = 0.0745      # flip-flop zone in HZ (0.048-0.098 AU)
BIPOLARIS_ECCENTRICITY = 0.05
BIPOLARIS_ROTATION_PERIOD = None    # set below (synchronous = orbital period)

# --- TLSR spin-orbit parameters ---
# Shakespeare & Steffen 2023 model: tidally locked with secular resonance
BIPOLARIS_TRIAXIALITY = 3e-5        # (B-A)/C — validated flip-flop regime (study5)
BIPOLARIS_TIDAL_Q = 22              # tidal quality factor — slushy/geologically active
# v3.4 flip-flop configuration (d=0.0745 AU, Q=20-25, triax=2-3e-5):
#   - TL episodes: 10-40 yr typical, occasionally 100+ yr
#   - PTB transitions: ~1 yr each, ~40% total time
#   - Both TL_ZERO and TL_PI occur irregularly
#   - Pattern: chaotic flip-flopping, may settle for decades then destabilize
# Q=20-25 is consistent with slushy/partially molten interior (not full ocean)

# --- Moon ---
# Debris moon (~350 Mya formation).
# Ceres-class body at 5 R_E, orbiting every ~18 hours.
MOON_RADIUS = 300e3                 # m (~300 km)
MOON_MASS = 3e-5 * M_EARTH          # kg (~Ceres-class)
MOON_DISTANCE = 5.0 * R_EARTH       # m — orbital distance
MOON_DISTANCE_PLANETARY_RADII = 5.0
MOON_PERIOD = 18.0 * 3600           # seconds (~18 hours)

# Aliases for backwards compatibility
NEXARA_RADIUS = MOON_RADIUS
NEXARA_MASS = MOON_MASS
NEXARA_DISTANCE = MOON_DISTANCE
NEXARA_DISTANCE_PLANETARY_RADII = MOON_DISTANCE_PLANETARY_RADII
NEXARA_PERIOD = MOON_PERIOD

# --- Planet d (outer HZ edge) ---
PLANET_D_DISTANCE_AU = 0.078
PLANET_D_MASS = 0.9 * M_EARTH
PLANET_D_RADIUS = 0.97 * R_EARTH
PLANET_D_ECCENTRICITY = 0.04

# --- Planet e (ejected ~500 Mya) ---
# Was at ~0.11 AU before ejection. No longer present.
PLANET_E_EJECTED = True
PLANET_E_ORIGINAL_DISTANCE_AU = 0.11
THE_BREAKING_MYA = 500              # Myr ago

# --- Atmosphere ---
ATMO_SURFACE_PRESSURE_ATM = 1.0     # approximate surface pressure
ATMO_N2_FRACTION = 0.91
ATMO_CO2_FRACTION_LOW = 0.002       # scenario: 0.2%
ATMO_CO2_FRACTION_MID = 0.006       # scenario: 0.6% (v3.2 balanced)
ATMO_CO2_FRACTION_HIGH = 0.010      # scenario: 1.0%
ATMO_O2_FRACTION = 0.02
ATMO_AR_FRACTION = 0.015
ATMO_CH4_FRACTION = 0.001           # trace methane (cold-side biosignature)
SURFACE_ALBEDO = 0.35               # Balanced with Q=20
SURFACE_GRAVITY = 8.7               # m/s^2 (1.0 M_E, 1.06 R_E) — ~0.89 g_E

# --- Temperature targets (EBM-validated at d=0.0745 AU) ---
# Actual: T_sub=397K, T_term=280K, T_anti=126K
TEMP_SUBSTELLAR = (370, 400)        # K — too hot for surface life
TEMP_TERMINATOR = (260, 290)        # K — habitable band
TEMP_ANTISTELLAR = (115, 135)       # K — frozen (cold traps may reach <112K)


# --- Derived quantities ---

def orbital_period(a_au: float, m_star_kg: float = STAR_MASS) -> float:
    """Kepler's third law: period in seconds from semi-major axis in AU."""
    a_m = a_au * AU
    return 2 * np.pi * np.sqrt(a_m**3 / (G * m_star_kg))


def orbital_period_days(a_au: float, m_star_kg: float = STAR_MASS) -> float:
    """Period in Earth days."""
    return orbital_period(a_au, m_star_kg) / DAY


def orbital_period_years(a_au: float, m_star_kg: float = STAR_MASS) -> float:
    """Period in Earth years."""
    return orbital_period(a_au, m_star_kg) / YEAR


def _kopparapu_seff(s_sun: float, coeffs: tuple, t_eff: float) -> float:
    """
    Effective stellar flux at HZ boundary (Kopparapu+ 2013 Table 3).
    S_eff = S_sun + a*T_star + b*T_star^2 + c*T_star^3 + d*T_star^4
    where T_star = T_eff - 5780 K.
    """
    t_star = t_eff - 5780.0
    a, b, c, d = coeffs
    return s_sun + a * t_star + b * t_star**2 + c * t_star**3 + d * t_star**4


# Kopparapu+ 2013 Table 3 coefficients (Runaway Greenhouse inner, Maximum Greenhouse outer)
_HZ_INNER_COEFFS = (1.0140e-4, 1.5860e-8, -4.3348e-12, -6.3430e-16)  # Runaway Greenhouse
_HZ_INNER_SSUN = 1.0385
_HZ_OUTER_COEFFS = (5.9578e-5, 1.6707e-9, -3.0058e-12, -5.1925e-16)  # Maximum Greenhouse
_HZ_OUTER_SSUN = 0.3507


def hz_inner_au(l_lsun: float = STAR_LUMINOSITY_LSUN, t_eff: float = STAR_TEFF) -> float:
    """Conservative inner edge of habitable zone (Kopparapu+ 2013, T_eff-dependent)."""
    s_eff = _kopparapu_seff(_HZ_INNER_SSUN, _HZ_INNER_COEFFS, t_eff)
    return np.sqrt(l_lsun / s_eff)


def hz_outer_au(l_lsun: float = STAR_LUMINOSITY_LSUN, t_eff: float = STAR_TEFF) -> float:
    """Conservative outer edge of habitable zone (Kopparapu+ 2013, T_eff-dependent)."""
    s_eff = _kopparapu_seff(_HZ_OUTER_SSUN, _HZ_OUTER_COEFFS, t_eff)
    return np.sqrt(l_lsun / s_eff)


def hill_radius(a_au: float, m_planet_kg: float, m_star_kg: float = STAR_MASS) -> float:
    """Hill radius in meters."""
    a_m = a_au * AU
    return a_m * (m_planet_kg / (3 * m_star_kg)) ** (1/3)


# Bipolaris orbital period
BIPOLARIS_PERIOD_DAYS = orbital_period_days(BIPOLARIS_DISTANCE_AU)
BIPOLARIS_PERIOD_YR = orbital_period_years(BIPOLARIS_DISTANCE_AU)

# Synchronous rotation: tidally locked to star
BIPOLARIS_ROTATION_PERIOD = orbital_period(BIPOLARIS_DISTANCE_AU)  # seconds

# Habitable zone bounds
HZ_INNER_AU = hz_inner_au()
HZ_OUTER_AU = hz_outer_au()

# Hill radius for moon stability check
BIPOLARIS_HILL_RADIUS_M = hill_radius(BIPOLARIS_DISTANCE_AU, BIPOLARIS_MASS)
BIPOLARIS_HILL_RADIUS_RE = BIPOLARIS_HILL_RADIUS_M / R_EARTH


if __name__ == "__main__":
    print("=== Bipolaris System Parameters (v2: Broken Chain) ===")
    print(f"Star: {STAR_SPECTRAL_TYPE}, {STAR_MASS_MSUN} M_sun, {STAR_LUMINOSITY_LSUN} L_sun")
    print(f"System age: {SYSTEM_AGE_GYR} Gyr")
    print(f"Habitable zone: {HZ_INNER_AU:.3f} - {HZ_OUTER_AU:.3f} AU")
    print()
    print("Planets:")
    for name, a, m, e in [
        ("b", PLANET_B_DISTANCE_AU, PLANET_B_MASS / M_EARTH, PLANET_B_ECCENTRICITY),
        ("c", PLANET_C_DISTANCE_AU, PLANET_C_MASS / M_EARTH, PLANET_C_ECCENTRICITY),
        ("Bipolaris", BIPOLARIS_DISTANCE_AU, BIPOLARIS_MASS / M_EARTH, BIPOLARIS_ECCENTRICITY),
        ("d", PLANET_D_DISTANCE_AU, PLANET_D_MASS / M_EARTH, PLANET_D_ECCENTRICITY),
    ]:
        p_days = orbital_period_days(a)
        in_hz = "HZ" if HZ_INNER_AU <= a <= HZ_OUTER_AU else ""
        print(f"  {name:10s}: a={a:.3f} AU, P={p_days:5.1f} d, M={m:.1f} M_E, e={e:.2f} {in_hz}")
    print()
    print(f"Bipolaris details:")
    print(f"  In HZ? {HZ_INNER_AU:.3f} <= {BIPOLARIS_DISTANCE_AU} <= {HZ_OUTER_AU:.3f}: "
          f"{'YES' if HZ_INNER_AU <= BIPOLARIS_DISTANCE_AU <= HZ_OUTER_AU else 'NO'}")
    print(f"  Rotation: synchronous ({BIPOLARIS_ROTATION_PERIOD / DAY:.1f} days)")
    print(f"  TLSR: triaxiality={BIPOLARIS_TRIAXIALITY:.0e}, Q={BIPOLARIS_TIDAL_Q}")
    print(f"  Hill radius: {BIPOLARIS_HILL_RADIUS_RE:.1f} R_E")
    print()
    print(f"Moon:")
    print(f"  Radius: {MOON_RADIUS/1e3:.0f} km")
    print(f"  Distance: {MOON_DISTANCE_PLANETARY_RADII:.0f} R_E "
          f"(stable limit ~{0.5*BIPOLARIS_HILL_RADIUS_RE:.0f} R_E)")
    print(f"  Period: {MOON_PERIOD/3600:.1f} hours")
    print()
    print(f"Companion: {COMPANION_MASS_MSUN} M_sun K5V at {COMPANION_DISTANCE_AU} AU, "
          f"incl={COMPANION_INCLINATION_DEG} deg")
    print()
    print(f"Instability: ~{THE_BREAKING_MYA} Mya (planet e ejected)")
