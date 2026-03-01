"""Unit tests for Bipolaris v2 physics calculations.

These tests validate that the v2 Broken Chain architecture parameters
produce physically consistent results.

System Design v2:
- M5.5V star (0.15 M_sun, 0.0022 L_sun)
- 4-planet quasi-resonant chain
- Bipolaris at 0.055 AU (middle of HZ)
- Elevated eccentricities from broken Laplace chain
"""

import numpy as np
import pytest

from shared.constants import (
    AU,
    BIPOLARIS_DISTANCE_AU,
    BIPOLARIS_ECCENTRICITY,
    BIPOLARIS_HILL_RADIUS_M,
    BIPOLARIS_HILL_RADIUS_RE,
    BIPOLARIS_MASS,
    BIPOLARIS_PERIOD_DAYS,
    BIPOLARIS_RADIUS,
    BIPOLARIS_ROTATION_PERIOD,
    BIPOLARIS_TIDAL_Q,
    BIPOLARIS_TRIAXIALITY,
    DAY,
    G,
    HZ_INNER_AU,
    HZ_OUTER_AU,
    M_EARTH,
    M_SUN,
    NEXARA_DISTANCE,
    NEXARA_DISTANCE_PLANETARY_RADII,
    PLANET_B_DISTANCE_AU,
    PLANET_C_DISTANCE_AU,
    PLANET_D_DISTANCE_AU,
    R_EARTH,
    STAR_MASS,
    STAR_MASS_MSUN,
    YEAR,
    hill_radius,
    hz_inner_au,
    hz_outer_au,
    orbital_period,
    orbital_period_days,
)


class TestV2OrbitalPeriods:
    """Verify orbital periods for v2 system with 0.15 M_sun star."""

    def test_bipolaris_period_order_of_magnitude(self) -> None:
        """Bipolaris at 0.055 AU around 0.15 M_sun should have P ~ 10-15 days."""
        p_days = orbital_period_days(BIPOLARIS_DISTANCE_AU)
        assert 10 < p_days < 15, f"Expected ~12 days, got {p_days:.1f}"

    def test_bipolaris_period_matches_precomputed(self) -> None:
        """Verify BIPOLARIS_PERIOD_DAYS is correctly computed."""
        expected = orbital_period_days(BIPOLARIS_DISTANCE_AU)
        assert BIPOLARIS_PERIOD_DAYS == pytest.approx(expected, rel=1e-10)

    def test_period_increases_with_distance(self) -> None:
        """Kepler's third: P ∝ a^(3/2), so outer planets have longer periods."""
        p_b = orbital_period_days(PLANET_B_DISTANCE_AU)
        p_c = orbital_period_days(PLANET_C_DISTANCE_AU)
        p_bipolaris = orbital_period_days(BIPOLARIS_DISTANCE_AU)
        p_d = orbital_period_days(PLANET_D_DISTANCE_AU)

        assert p_b < p_c < p_bipolaris < p_d

    def test_period_ratios_quasi_resonant(self) -> None:
        """Adjacent period ratios should be near simple resonances.

        Broken chain: ratios are near but not exactly at 2:1, 3:2, etc.
        From system design: b:c = 2.3, c:Bip = 2.3, Bip:d = 1.7
        """
        p_b = orbital_period(PLANET_B_DISTANCE_AU)
        p_c = orbital_period(PLANET_C_DISTANCE_AU)
        p_bipolaris = orbital_period(BIPOLARIS_DISTANCE_AU)
        p_d = orbital_period(PLANET_D_DISTANCE_AU)

        ratio_cb = p_c / p_b
        ratio_bc = p_bipolaris / p_c
        ratio_db = p_d / p_bipolaris

        # Near resonant: ratios should be near simple integers
        # b:c and c:Bip near 2:1, Bip:d near 5:3
        for ratio, name, expected in [
            (ratio_cb, "c/b", 2.0),
            (ratio_bc, "Bipolaris/c", 2.0),
            (ratio_db, "d/Bipolaris", 1.67),
        ]:
            assert 1.2 < ratio < 2.5, f"Period ratio {name} = {ratio:.3f} out of range"
            # Check it's near a simple resonance
            assert abs(ratio - expected) < 0.5, (
                f"Period ratio {name} = {ratio:.3f} not near expected {expected}"
            )

    def test_synchronous_rotation_equals_orbital(self) -> None:
        """Synchronous rotation period should equal orbital period."""
        p_orbital = orbital_period(BIPOLARIS_DISTANCE_AU)
        assert BIPOLARIS_ROTATION_PERIOD == pytest.approx(p_orbital, rel=1e-10)


class TestV2TidalParameters:
    """Tests for v2 tidal dissipation parameters."""

    def test_tidal_epsilon_positive(self) -> None:
        """Tidal epsilon should be positive for physical parameters."""
        from tlsr_spin.physics import tidal_epsilon

        n = 2 * np.pi / BIPOLARIS_ROTATION_PERIOD  # synchronous spin
        eps = tidal_epsilon(
            m_star=STAR_MASS,
            m_planet=BIPOLARIS_MASS,
            r_planet=BIPOLARIS_RADIUS,
            a=BIPOLARIS_DISTANCE_AU * AU,
            tidal_q=BIPOLARIS_TIDAL_Q,
            omega_spin=n,
        )
        assert eps > 0

    def test_tidal_epsilon_v2_stronger_than_old(self) -> None:
        """At 0.055 AU, tidal epsilon should be ~10x higher than at 0.18 AU.

        ε ∝ a^(-6), so (0.18/0.055)^6 ≈ 1500x stronger.
        """
        from tlsr_spin.physics import tidal_epsilon

        # Common parameters
        kwargs = dict(
            m_star=STAR_MASS,
            m_planet=BIPOLARIS_MASS,
            r_planet=BIPOLARIS_RADIUS,
            tidal_q=BIPOLARIS_TIDAL_Q,
        )

        # v2 at 0.055 AU
        a_v2 = BIPOLARIS_DISTANCE_AU * AU
        n_v2 = np.sqrt(G * STAR_MASS / a_v2**3)
        eps_v2 = tidal_epsilon(**kwargs, a=a_v2, omega_spin=n_v2)

        # Old design at 0.18 AU
        a_old = 0.18 * AU
        n_old = np.sqrt(G * STAR_MASS / a_old**3)
        eps_old = tidal_epsilon(**kwargs, a=a_old, omega_spin=n_old)

        # Ratio should be large (ε ∝ a^(-6) primarily)
        ratio = eps_v2 / eps_old
        assert ratio > 100, f"Expected ε ratio > 100, got {ratio:.1f}"

    def test_tidal_q_scaling(self) -> None:
        """ε(Q=10) / ε(Q=100) = 10 (inverse proportionality)."""
        from tlsr_spin.physics import tidal_epsilon

        a_m = BIPOLARIS_DISTANCE_AU * AU
        n = np.sqrt(G * STAR_MASS / a_m**3)

        kwargs = dict(
            m_star=STAR_MASS,
            m_planet=BIPOLARIS_MASS,
            r_planet=BIPOLARIS_RADIUS,
            a=a_m,
            omega_spin=n,
        )

        eps_q10 = tidal_epsilon(**kwargs, tidal_q=10)
        eps_q100 = tidal_epsilon(**kwargs, tidal_q=100)

        assert eps_q10 == pytest.approx(10 * eps_q100, rel=1e-10)

    def test_tidal_timescale_reasonable(self) -> None:
        """Tidal timescale τ = 1/ε should be in years, not seconds or Gyr.

        For v2 at 0.055 AU with Q=10, expect τ ~ 10^3-10^6 years.
        """
        from tlsr_spin.physics import tidal_epsilon

        a_m = BIPOLARIS_DISTANCE_AU * AU
        n = np.sqrt(G * STAR_MASS / a_m**3)

        eps = tidal_epsilon(
            m_star=STAR_MASS,
            m_planet=BIPOLARIS_MASS,
            r_planet=BIPOLARIS_RADIUS,
            a=a_m,
            tidal_q=BIPOLARIS_TIDAL_Q,
            omega_spin=n,
        )

        tau_yr = (1.0 / eps) / YEAR
        assert 1e2 < tau_yr < 1e7, f"Tidal timescale {tau_yr:.0e} yr out of range"


class TestV2HabitableZone:
    """Tests for v2 habitable zone calculations."""

    def test_hz_bounds_order(self) -> None:
        """Inner HZ boundary should be less than outer."""
        assert HZ_INNER_AU < HZ_OUTER_AU

    def test_hz_inner_reasonable(self) -> None:
        """For M5.5V (0.0022 L_sun), inner HZ should be ~0.04-0.06 AU."""
        inner = hz_inner_au()
        assert 0.03 < inner < 0.07, f"HZ inner = {inner:.3f} AU out of range"

    def test_hz_outer_reasonable(self) -> None:
        """For M5.5V, outer HZ should be ~0.08-0.12 AU."""
        outer = hz_outer_au()
        assert 0.07 < outer < 0.15, f"HZ outer = {outer:.3f} AU out of range"

    def test_bipolaris_in_hz(self) -> None:
        """Bipolaris at 0.055 AU should be inside the habitable zone."""
        assert HZ_INNER_AU <= BIPOLARIS_DISTANCE_AU <= HZ_OUTER_AU, (
            f"Bipolaris at {BIPOLARIS_DISTANCE_AU} AU "
            f"not in HZ [{HZ_INNER_AU:.3f}, {HZ_OUTER_AU:.3f}]"
        )

    def test_planet_b_inside_hz_inner(self) -> None:
        """Planet b at 0.018 AU should be inside (too hot) the HZ."""
        assert PLANET_B_DISTANCE_AU < HZ_INNER_AU

    def test_planet_d_at_hz_edge(self) -> None:
        """Planet d at 0.078 AU should be near outer edge of HZ."""
        # d should still be in HZ but near edge
        assert HZ_INNER_AU < PLANET_D_DISTANCE_AU < HZ_OUTER_AU


class TestV2HillRadius:
    """Tests for Hill radius and moon stability."""

    def test_hill_radius_calculation(self) -> None:
        """Verify Hill radius formula: R_H = a (m_planet / 3 m_star)^(1/3)."""
        r_h = hill_radius(BIPOLARIS_DISTANCE_AU, BIPOLARIS_MASS)

        # Manual calculation
        a_m = BIPOLARIS_DISTANCE_AU * AU
        expected = a_m * (BIPOLARIS_MASS / (3 * STAR_MASS)) ** (1 / 3)

        assert r_h == pytest.approx(expected, rel=1e-10)

    def test_hill_radius_precomputed(self) -> None:
        """Verify precomputed BIPOLARIS_HILL_RADIUS_M is correct."""
        expected = hill_radius(BIPOLARIS_DISTANCE_AU, BIPOLARIS_MASS)
        assert BIPOLARIS_HILL_RADIUS_M == pytest.approx(expected, rel=1e-10)

    def test_hill_radius_planetary_radii(self) -> None:
        """Verify conversion to planetary radii."""
        expected = BIPOLARIS_HILL_RADIUS_M / R_EARTH
        assert BIPOLARIS_HILL_RADIUS_RE == pytest.approx(expected, rel=1e-10)

    def test_nexara_inside_stable_zone(self) -> None:
        """Moon at 5 R_E should be inside the stable zone (< 0.5 R_H).

        For prograde moons, stability limit is ~0.5 R_H.
        """
        nexara_distance_m = NEXARA_DISTANCE
        stable_limit = 0.5 * BIPOLARIS_HILL_RADIUS_M

        assert nexara_distance_m < stable_limit, (
            f"Moon at {nexara_distance_m / R_EARTH:.1f} R_E "
            f"> stable limit {stable_limit / R_EARTH:.1f} R_E"
        )

    def test_nexara_distance_planetary_radii(self) -> None:
        """Verify Moon distance is correctly specified as 5 R_E."""
        assert NEXARA_DISTANCE_PLANETARY_RADII == 5.0
        assert NEXARA_DISTANCE == pytest.approx(5.0 * R_EARTH, rel=1e-10)


class TestV2LibrationFrequency:
    """Tests for libration frequency ω_s calculations."""

    def test_omega_s_squared_positive(self) -> None:
        """ω_s² should be positive for baseline parameters."""
        from tlsr_spin.physics import omega_s_squared

        n = 2 * np.pi / BIPOLARIS_ROTATION_PERIOD
        w_s2 = omega_s_squared(n, BIPOLARIS_TRIAXIALITY, BIPOLARIS_ECCENTRICITY)
        assert w_s2 > 0

    def test_libration_period_reasonable(self) -> None:
        """Libration period P_lib = 2π/ω_s should be on order of years.

        For tidally locked planets, typical libration periods are
        years to decades.
        """
        from tlsr_spin.physics import omega_s_squared

        n = 2 * np.pi / BIPOLARIS_ROTATION_PERIOD
        w_s2 = omega_s_squared(n, BIPOLARIS_TRIAXIALITY, BIPOLARIS_ECCENTRICITY)
        w_s = np.sqrt(w_s2)
        p_lib_yr = (2 * np.pi / w_s) / YEAR

        assert 0.1 < p_lib_yr < 100, f"Libration period {p_lib_yr:.1f} yr out of range"

    def test_libration_period_triax_sweep(self) -> None:
        """Libration period should scale with triaxiality: P ∝ 1/sqrt(triax)."""
        from tlsr_spin.physics import omega_s_squared

        n = 2 * np.pi / BIPOLARIS_ROTATION_PERIOD
        e = BIPOLARIS_ECCENTRICITY

        # Compute for different triaxialities
        triaxialities = [1e-5, 1e-4, 1e-3]
        periods = []
        for triax in triaxialities:
            w_s2 = omega_s_squared(n, triax, e)
            w_s = np.sqrt(w_s2)
            p_lib_yr = (2 * np.pi / w_s) / YEAR
            periods.append(p_lib_yr)

        # Check scaling: P ∝ triax^(-1/2)
        # P(1e-4) / P(1e-3) should be sqrt(10) ≈ 3.16
        ratio = periods[1] / periods[2]
        assert ratio == pytest.approx(np.sqrt(10), rel=0.01)
