"""
Parameter exploration scenarios for the Bipolaris system.

System Design v2: Broken Chain Architecture

Scenarios for exploring TLSR dynamics in the new architecture. The baseline
is now a 4-planet quasi-resonant chain around an M5.5V star, with elevated
eccentricities from the broken Laplace chain.

Scenarios:
  A: Baseline — v2 architecture with current eccentricities
  B: Stable Era — pre-Breaking configuration (5 planets, lower e)
  C: High-Q — Q=100, weaker tidal dissipation
  D: Low triaxiality — (B-A)/C = 1e-5, weaker restoring torque
  E: High triaxiality — (B-A)/C = 1e-3, stronger restoring torque
  F: TRAPPIST-1 f (validation) — complete planet f parameters
  G: TRAPPIST-1 star, sweepable distance — for mapping
"""

import shared.constants as _constants_module
from shared.constants import (
    BIPOLARIS_DISTANCE_AU,
    BIPOLARIS_ECCENTRICITY,
    BIPOLARIS_MASS,
    BIPOLARIS_RADIUS,
    L_SUN,
    M_EARTH,
    M_JUPITER,
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
    R_EARTH,
    STAR_LUMINOSITY_LSUN,
    STAR_MASS_MSUN,
    STAR_TEFF,
)

# Valid override keys: all uppercase names in constants module
_VALID_KEYS = {name for name in dir(_constants_module)
               if name.isupper() and not name.startswith("_")}

SCENARIOS = {
    "A": {
        "label": "Baseline (v2 Broken Chain)",
        "description": (
            "Current architecture: 4-planet quasi-resonant chain around M5.5V. "
            "Elevated eccentricities from broken Laplace chain. Default TLSR parameters."
        ),
        "overrides": {},
    },
    "B": {
        "label": "Stable Era (pre-instability)",
        "description": (
            "Configuration before the instability event: 5-planet Laplace chain with "
            "lower eccentricities. Represents 7+ Gyr of stable tidal locking."
        ),
        "overrides": {
            # Lower eccentricities (resonance-damped)
            "PLANET_B_ECCENTRICITY": 0.01,
            "PLANET_C_ECCENTRICITY": 0.01,
            "BIPOLARIS_ECCENTRICITY": 0.01,
            "PLANET_D_ECCENTRICITY": 0.01,
            # Planet e still present
            "PLANET_E_EJECTED": False,
        },
    },
    "C": {
        "label": "High-Q (desiccated)",
        "description": "Q=100: weaker tidal dissipation, longer regime transitions",
        "overrides": {
            "BIPOLARIS_TIDAL_Q": 100,
        },
    },
    "D": {
        "label": "Low triaxiality",
        "description": "(B-A)/C = 1e-5: weaker restoring torque, more PTB expected",
        "overrides": {
            "BIPOLARIS_TRIAXIALITY": 1e-5,
        },
    },
    "E": {
        "label": "High triaxiality",
        "description": "(B-A)/C = 1e-3: stronger restoring torque, more locking expected",
        "overrides": {
            "BIPOLARIS_TRIAXIALITY": 1e-3,
        },
    },
    "F": {
        "label": "TRAPPIST-1 f (validation)",
        "description": (
            "Complete TRAPPIST-1 f parameters for validation against "
            "Shakespeare & Steffen 2023. Use for cross-checking dynamics."
        ),
        "overrides": {
            "STAR_MASS_MSUN": 0.0898,
            "STAR_MASS": 0.0898 * M_SUN,
            "STAR_LUMINOSITY_LSUN": 0.000553,
            "STAR_LUMINOSITY": 0.000553 * L_SUN,
            "STAR_TEFF": 2566,
            "BIPOLARIS_DISTANCE_AU": 0.03853,  # TRAPPIST-1 f semi-major axis
            "BIPOLARIS_MASS": 1.039 * M_EARTH,
            "BIPOLARIS_RADIUS": 1.045 * R_EARTH,
            "BIPOLARIS_TIDAL_Q": 10,
        },
    },
    "G": {
        "label": "TRAPPIST-1 star, sweepable distance",
        "description": (
            "TRAPPIST-1 stellar properties, semi-major axis specified via "
            "CLI --a-override. For goldilocks zone mapping."
        ),
        "overrides": {
            "STAR_MASS_MSUN": 0.0898,
            "STAR_MASS": 0.0898 * M_SUN,
            "STAR_LUMINOSITY_LSUN": 0.000553,
            "STAR_LUMINOSITY": 0.000553 * L_SUN,
            "STAR_TEFF": 2566,
        },
    },
    "H": {
        "label": "Adjusted (v2.1)",
        "description": (
            "Adjusted parameters: moved outward to 0.068 AU with high albedo 0.45. "
            "Best compromise between cooler temps and tidal lock capability. "
            "T_sub~380K, T_term~270K, T_anti~120K. Instability at 500 Mya."
        ),
        "overrides": {
            "BIPOLARIS_DISTANCE_AU": 0.068,
            "ATMO_CO2_FRACTION_MID": 0.003,
            "SURFACE_ALBEDO": 0.45,
            "THE_BREAKING_MYA": 500,
        },
    },
    "I": {
        "label": "High-PTB (v3.1)",
        "description": (
            "HIGH-PTB: 0.070 AU, A=0.30, CO2=0.5%, triax=3e-5, Q=8. "
            "Maximizes PTB (75.4% PTB / 25% TL_ZERO / 0% TL_PI). "
            "T_sub=412K, T_term=291K, T_anti=131K, cold_trap=96K. "
            "Use when PTB-dominated dynamics are desired."
        ),
        "overrides": {
            "SURFACE_ALBEDO": 0.30,
            "ATMO_CO2_FRACTION_MID": 0.005,
            "BIPOLARIS_TIDAL_Q": 8,
        },
    },
    "J": {
        "label": "Balanced Regimes (v3.2)",
        "description": (
            "BALANCED: 0.070 AU, A=0.35, CO2=0.6%, triax=3e-5, Q=20. "
            "Near-equal regime split (36% PTB / 30% TL_ZERO / 34% TL_PI). "
            "Long stable tidal lock periods with intermittent PTB transitions. "
            "T_sub=409K, T_term=289K, T_anti=130K, cold_trap=95K. "
            "Default configuration."
        ),
        "overrides": {
            "SURFACE_ALBEDO": 0.35,
            "ATMO_CO2_FRACTION_MID": 0.006,
            "BIPOLARIS_TIDAL_Q": 20,
        },
    },
}


def get_scenario(name: str) -> dict:
    """Return a dict of constant overrides for the named scenario.

    Usage:
        overrides = get_scenario("A")
        star_mass = overrides.get("STAR_MASS_MSUN", STAR_MASS_MSUN)

    Args:
        name: Scenario key (A, B, C, D, E, F, G).

    Returns:
        Dict of constant name -> override value.

    Raises:
        ValueError: If scenario name is unknown or has invalid keys.
    """
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{name}'. Choose from: {list(SCENARIOS.keys())}")
    overrides = SCENARIOS[name]["overrides"]
    bad_keys = set(overrides.keys()) - _VALID_KEYS
    if bad_keys:
        raise ValueError(
            f"Scenario '{name}' has invalid override keys: {bad_keys}. "
            f"These don't match any constant in shared.constants."
        )
    return overrides


def get_scenario_label(name: str) -> str:
    """Human-readable label for a scenario."""
    return SCENARIOS[name]["label"]


def apply_overrides(module_constants: dict, overrides: dict) -> dict:
    """Given base constant values and overrides, return merged dict.

    Args:
        module_constants: Base constant values.
        overrides: Override values from a scenario.

    Returns:
        Merged dict with overrides applied.
    """
    merged = dict(module_constants)
    merged.update(overrides)
    return merged


def scenario_summary() -> None:
    """Print a summary table of all scenarios."""
    print(f"{'Name':<6} {'Label':<35} {'Star M_sun':<12} {'a (AU)':<10}")
    print("-" * 63)
    for name, s in SCENARIOS.items():
        o = s["overrides"]
        print(f"{name:<6} {s['label']:<35} "
              f"{o.get('STAR_MASS_MSUN', STAR_MASS_MSUN):<12.4f} "
              f"{o.get('BIPOLARIS_DISTANCE_AU', BIPOLARIS_DISTANCE_AU):<10.3f}")


if __name__ == "__main__":
    scenario_summary()
