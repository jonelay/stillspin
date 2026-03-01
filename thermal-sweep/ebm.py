"""
Energy Balance Model for tidally-locked exoplanet temperature estimation.

Simple 3-zone model:
- Substellar: direct insolation, peak temperature
- Terminator: grazing incidence, intermediate temperature
- Antistellar: no direct flux, radiative loss only

Greenhouse parameterization follows grey-atmosphere approximation with
CO₂-dependent optical depth.
"""

import numpy as np

# Physical constants
SIGMA = 5.670374e-8  # Stefan-Boltzmann (W m^-2 K^-4)
L_SUN = 3.828e26     # Solar luminosity (W)
AU = 1.496e11        # Astronomical unit (m)


def stellar_flux(distance_au: float, luminosity_lsun: float) -> float:
    """Calculate stellar flux at given distance.

    Args:
        distance_au: Distance from star in AU.
        luminosity_lsun: Stellar luminosity in solar units.

    Returns:
        Flux in W/m^2.
    """
    l_star = luminosity_lsun * L_SUN
    d = distance_au * AU
    return l_star / (4 * np.pi * d**2)


def equilibrium_temperature(flux: float, albedo: float) -> float:
    """Calculate equilibrium temperature (no atmosphere).

    Uses instantaneous re-radiation (tidally locked, no heat transport).

    Args:
        flux: Incident stellar flux in W/m^2.
        albedo: Surface bond albedo (0-1).

    Returns:
        Equilibrium temperature in K.
    """
    return ((flux * (1 - albedo)) / SIGMA) ** 0.25


def greenhouse_optical_depth(co2_fraction: float, pressure_atm: float = 1.0) -> float:
    """Estimate optical depth from CO₂ fraction.

    Parameterized as logarithmic scaling from Earth's value.
    Includes pressure broadening factor.

    Args:
        co2_fraction: CO₂ mixing ratio (e.g., 0.01 for 1%).
        pressure_atm: Surface pressure in Earth atmospheres.

    Returns:
        Effective optical depth τ.
    """
    tau_earth = 1.0
    co2_earth = 0.0004
    tau = tau_earth * np.log(1 + co2_fraction / co2_earth) / np.log(1 + 1.0)
    return tau * (pressure_atm ** 0.5)


def surface_temperature(t_eq: float, tau: float) -> float:
    """Apply greenhouse warming to equilibrium temperature.

    Grey-atmosphere approximation: T_surf = T_eq × (1 + 0.75τ)^0.25

    Args:
        t_eq: Equilibrium temperature (K).
        tau: Optical depth.

    Returns:
        Surface temperature (K).
    """
    return t_eq * (1 + 0.75 * tau) ** 0.25


def tidal_lock_temperatures(
    distance_au: float,
    luminosity_lsun: float,
    albedo: float,
    co2_fraction: float,
    pressure_atm: float = 1.0,
) -> dict:
    """Calculate temperature distribution for tidally locked planet.

    Three-zone model:
    - Substellar: Full insolation (μ = 1)
    - Terminator: Grazing incidence (effective μ ~ 0.3)
    - Antistellar: No direct flux, radiative equilibrium with heat transport

    Args:
        distance_au: Distance from star in AU.
        luminosity_lsun: Stellar luminosity in solar units.
        albedo: Surface bond albedo.
        co2_fraction: CO₂ mixing ratio.
        pressure_atm: Surface pressure in Earth atmospheres.

    Returns:
        Dict with temperatures for each zone and intermediate values.
    """
    flux = stellar_flux(distance_au, luminosity_lsun)
    tau = greenhouse_optical_depth(co2_fraction, pressure_atm)

    # Substellar point: full insolation
    t_eq_sub = equilibrium_temperature(flux, albedo)
    t_substellar = surface_temperature(t_eq_sub, tau)

    # Terminator zone: grazing incidence
    # Effective flux ~ flux × cos(80°) ≈ 0.17, but with heat transport ~0.3
    flux_term = flux * 0.25
    t_eq_term = equilibrium_temperature(flux_term, albedo)
    t_terminator = surface_temperature(t_eq_term, tau)

    # Antistellar: no direct flux, only transported heat and radiative loss
    # Tidally locked planets with thin atmospheres have very cold night sides.
    # Heat transport efficiency depends on atmospheric thickness and dynamics.
    # For ~1 atm N2-dominated atmosphere, transport is weak.
    # Use empirical scaling: T_anti ~ T_term × transport_factor
    # Transport factor ~0.4-0.5 for thin atmo (cf. GCM results for tidally locked)
    transport_factor = 0.45
    t_antistellar = t_terminator * transport_factor

    return {
        "flux": flux,
        "tau": tau,
        "t_eq_substellar": t_eq_sub,
        "t_substellar": t_substellar,
        "t_terminator": t_terminator,
        "t_antistellar": t_antistellar,
    }
