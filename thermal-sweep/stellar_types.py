"""Stellar parameter database for coarse thermal sweep.

Provides M-dwarf stellar parameters and HZ calculation utilities.
"""

import numpy as np

# Physical constants
SIGMA = 5.670374e-8  # Stefan-Boltzmann (W m^-2 K^-4)
L_SUN = 3.828e26     # Solar luminosity (W)
AU = 1.496e11        # Astronomical unit (m)

STELLAR_TYPES = {
    "M5.5V": {
        "mass": 0.15,       # M_sun
        "luminosity": 0.0022,  # L_sun
        "teff": 2900,       # K
        "description": "Current baseline star"
    },
    "M6.5V": {
        "mass": 0.11,
        "luminosity": 0.0010,
        "teff": 2750,
        "description": "Intermediate coolness"
    },
    "M8V": {
        "mass": 0.090,
        "luminosity": 0.00055,
        "teff": 2566,
        "description": "TRAPPIST-1 analog"
    },
}


def compute_hz_bounds(luminosity_lsun: float, teff: float) -> tuple[float, float]:
    """Compute habitable zone boundaries using simple luminosity scaling.

    Conservative HZ limits based on stellar luminosity, using empirical
    boundaries for M-dwarfs.

    Args:
        luminosity_lsun: Stellar luminosity in solar units.
        teff: Stellar effective temperature in K.

    Returns:
        Tuple of (inner_au, outer_au) habitable zone boundaries.
    """
    # Simple luminosity-based HZ (empirical for M-dwarfs)
    # Inner: runaway greenhouse (~0.95 Earth flux)
    # Outer: maximum greenhouse (~0.35 Earth flux)
    inner_au = np.sqrt(luminosity_lsun / 1.05)
    outer_au = np.sqrt(luminosity_lsun / 0.30)

    return inner_au, outer_au


def sample_hz_positions(hz_inner: float, hz_outer: float, n: int = 3) -> list[float]:
    """Sample evenly-spaced positions in the habitable zone.

    Args:
        hz_inner: Inner HZ boundary in AU.
        hz_outer: Outer HZ boundary in AU.
        n: Number of positions to sample (default: 3).

    Returns:
        List of distances in AU.
    """
    return [hz_inner + i * (hz_outer - hz_inner) / (n - 1) for i in range(n)]


def stellar_type_summary() -> None:
    """Print summary table of stellar types and their HZ."""
    print(f"{'Type':<8} {'M (M☉)':<10} {'L (L☉)':<12} {'T_eff (K)':<12} {'HZ inner':<12} {'HZ outer':<12}")
    print("-" * 66)
    for name, params in STELLAR_TYPES.items():
        hz_inner, hz_outer = compute_hz_bounds(params["luminosity"], params["teff"])
        print(f"{name:<8} {params['mass']:<10.3f} {params['luminosity']:<12.5f} "
              f"{params['teff']:<12} {hz_inner:<12.4f} {hz_outer:<12.4f}")


if __name__ == "__main__":
    stellar_type_summary()
