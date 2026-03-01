"""Shared pytest fixtures for TLSR spin dynamics tests.

System Design v2: Broken Chain Architecture
- 4-planet quasi-resonant chain around M5.5V star
- Elevated eccentricities from broken Laplace chain
- Bipolaris in middle of HZ at 0.055 AU
"""

import numpy as np
import pytest

from shared.constants import (
    AU,
    BIPOLARIS_DISTANCE_AU,
    BIPOLARIS_ECCENTRICITY,
    BIPOLARIS_MASS,
    BIPOLARIS_RADIUS,
    STAR_MASS,
    YEAR,
    orbital_period,
)
from shared.scenarios import get_scenario


# --- Time constants for test runs ---
N_ORBITS_QUICK = 1_000
N_ORBITS_MEDIUM = 100_000
N_ORBITS_LONG = 1_000_000
N_ORBITS_STABILITY = 10_000_000


# --- Pytest markers ---
def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: test takes > 10 seconds")
    config.addinivalue_line("markers", "integration: pipeline tests")
    config.addinivalue_line("markers", "scenario: scenario comparison tests")
    config.addinivalue_line("markers", "stability: long-duration tests (hours)")


# --- System fixtures ---
@pytest.fixture
def v2_simulation() -> dict:
    """Build Bipolaris system simulation parameters from constants.

    Returns dict with all parameters needed for spin dynamics.
    """
    from tlsr_spin.nbody import build_bipolaris_system

    sim = build_bipolaris_system()

    # Get orbital parameters for Bipolaris (planet index 3)
    orb = sim.particles[3].orbit()
    p_orbital_s = orb.P * YEAR  # convert from sim units (years) to seconds

    return {
        "sim": sim,
        "m_star": STAR_MASS,
        "m_planet": BIPOLARIS_MASS,
        "r_planet": BIPOLARIS_RADIUS,
        "a_au": BIPOLARIS_DISTANCE_AU,
        "a_m": BIPOLARIS_DISTANCE_AU * AU,
        "e": BIPOLARIS_ECCENTRICITY,
        "p_orbital_s": p_orbital_s,
        "n_mean": 2 * np.pi / p_orbital_s,  # rad/s
    }


@pytest.fixture
def scenario_a() -> dict:
    """Scenario A: Baseline v2 Broken Chain (current era)."""
    return get_scenario("A")


@pytest.fixture
def scenario_b() -> dict:
    """Scenario B: Stable Era pre-Breaking (lower eccentricities)."""
    return get_scenario("B")


@pytest.fixture
def scenario_c() -> dict:
    """Scenario C: High-Q (Q=100, weaker tidal dissipation)."""
    return get_scenario("C")


@pytest.fixture
def scenario_d() -> dict:
    """Scenario D: Low triaxiality ((B-A)/C = 1e-5)."""
    return get_scenario("D")


@pytest.fixture
def scenario_e() -> dict:
    """Scenario E: High triaxiality ((B-A)/C = 1e-3)."""
    return get_scenario("E")


@pytest.fixture
def scenario_f() -> dict:
    """Scenario F: TRAPPIST-1 f validation."""
    return get_scenario("F")


@pytest.fixture
def scenario_g() -> dict:
    """Scenario G: TRAPPIST-1 star, sweepable distance."""
    return get_scenario("G")


@pytest.fixture
def short_nbody_result() -> dict:
    """Pre-computed 1k orbit N-body for fast tests.

    Returns dict with t, e, n, a arrays from a quick integration.
    """
    from tlsr_spin.nbody import build_bipolaris_system, integrate_and_extract

    sim = build_bipolaris_system()
    result = integrate_and_extract(
        sim,
        planet_idx=3,  # Bipolaris
        n_orbits=N_ORBITS_QUICK,
        samples_per_orbit=10,
    )
    return result


@pytest.fixture
def medium_nbody_result() -> dict:
    """Pre-computed 100k orbit N-body for integration tests.

    Returns dict with t, e, n, a arrays.
    Marked slow since it takes ~30-60 seconds.
    """
    from tlsr_spin.nbody import build_bipolaris_system, integrate_and_extract

    sim = build_bipolaris_system()
    result = integrate_and_extract(
        sim,
        planet_idx=3,
        n_orbits=N_ORBITS_MEDIUM,
        samples_per_orbit=10,
    )
    return result


@pytest.fixture
def short_spin_result(short_nbody_result: dict) -> dict:
    """Pre-computed spin integration for quick tests.

    Uses baseline triaxiality and Q values.
    """
    from shared.constants import BIPOLARIS_TIDAL_Q, BIPOLARIS_TRIAXIALITY

    from tlsr_spin.spin_integrator import integrate_spin

    nbody = short_nbody_result
    a_mean = float(np.mean(nbody["a"])) * AU

    return integrate_spin(
        times=nbody["t"],
        e_t=nbody["e"],
        n_t=nbody["n"],
        m_star=STAR_MASS,
        m_planet=BIPOLARIS_MASS,
        r_planet=BIPOLARIS_RADIUS,
        a_mean=a_mean,
        tidal_q=BIPOLARIS_TIDAL_Q,
        triaxiality=BIPOLARIS_TRIAXIALITY,
    )


@pytest.fixture
def short_regime_result(short_spin_result: dict) -> "RegimeResult":
    """Pre-computed regime classification for quick tests."""
    from tlsr_spin.regime_classifier import classify_regimes

    return classify_regimes(
        short_spin_result["t"],
        short_spin_result["gamma"],
    )


# --- Helper fixtures ---
@pytest.fixture
def default_triaxiality() -> float:
    """Default triaxiality (B-A)/C from constants."""
    from shared.constants import BIPOLARIS_TRIAXIALITY

    return BIPOLARIS_TRIAXIALITY


@pytest.fixture
def default_tidal_q() -> int:
    """Default tidal Q from constants."""
    from shared.constants import BIPOLARIS_TIDAL_Q

    return BIPOLARIS_TIDAL_Q


@pytest.fixture
def hz_bounds() -> tuple[float, float]:
    """Conservative habitable zone bounds in AU."""
    from shared.constants import HZ_INNER_AU, HZ_OUTER_AU

    return (HZ_INNER_AU, HZ_OUTER_AU)


# Type annotation for RegimeResult
if False:  # TYPE_CHECKING
    from tlsr_spin.regime_classifier import RegimeResult
