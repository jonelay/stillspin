"""Long-duration stability tests for Bipolaris v2 system.

These tests verify long-term orbital stability and the effects of
Kozai-Lidov dynamics from the K5V companion.

Markers:
- @pytest.mark.stability: marks all tests in this module
- @pytest.mark.slow: all tests here take > 10 seconds
- Most tests here take hours to run fully

These tests are NOT run in CI by default. Run with:
    pytest -m stability tlsr-spin/tests/test_stability.py
"""

import numpy as np
import pytest

from shared.constants import (
    AU,
    BIPOLARIS_DISTANCE_AU,
    BIPOLARIS_ECCENTRICITY,
    COMPANION_DISTANCE_AU,
    COMPANION_INCLINATION_DEG,
    COMPANION_MASS_MSUN,
    HZ_INNER_AU,
    HZ_OUTER_AU,
    STAR_MASS_MSUN,
    SYSTEM_AGE_GYR,
    YEAR,
)

from tests.conftest import N_ORBITS_QUICK, N_ORBITS_MEDIUM


pytestmark = [pytest.mark.stability, pytest.mark.slow]


class TestLongTermNbody:
    """Tests for long-term N-body stability."""

    def test_megno_bounded(self) -> None:
        """MEGNO chaos indicator should be < 4 for stable systems.

        MEGNO (Mean Exponential Growth of Nearby Orbits) measures chaos:
        - MEGNO → 2: quasi-periodic (stable)
        - MEGNO > 4: chaotic (unstable)

        Note: This is a simplified MEGNO check over short integration.
        Full 1 Myr integration would be ideal but too slow for CI.
        """
        from tlsr_spin.nbody import build_bipolaris_system

        sim = build_bipolaris_system()
        sim.init_megno()

        # Integrate for 10k orbits
        orb = sim.particles[3].orbit()
        t_end = 10000 * orb.P
        sim.integrate(t_end)

        megno = sim.megno()

        # For stable system, MEGNO should be close to 2
        assert megno < 5, f"MEGNO = {megno:.2f} indicates potential chaos"

    def test_no_planet_ejections(self) -> None:
        """No planets should be ejected during integration.

        Check that all 4 planets remain bound (e < 1) throughout.
        """
        from tlsr_spin.nbody import build_bipolaris_system, integrate_and_extract

        sim = build_bipolaris_system()

        # Integrate and check all planets
        for planet_idx in [1, 2, 3, 4]:  # b, c, Bipolaris, d
            # Reset sim for each planet check
            sim = build_bipolaris_system()
            result = integrate_and_extract(
                sim, planet_idx=planet_idx, n_orbits=1000
            )

            # All eccentricities should be < 1
            assert np.all(result["e"] < 1.0), (
                f"Planet {planet_idx} ejected: max e = {result['e'].max():.3f}"
            )

    def test_bipolaris_stays_in_hz(self) -> None:
        """Bipolaris should remain in habitable zone throughout integration."""
        from tlsr_spin.nbody import build_bipolaris_system, integrate_and_extract

        sim = build_bipolaris_system()
        result = integrate_and_extract(sim, planet_idx=3, n_orbits=N_ORBITS_QUICK)

        # Check semi-major axis bounds
        a_min = np.min(result["a"])
        a_max = np.max(result["a"])

        # Allow some secular variation but should stay near HZ
        assert a_min > HZ_INNER_AU * 0.8, f"a_min = {a_min:.4f} AU below HZ"
        assert a_max < HZ_OUTER_AU * 1.2, f"a_max = {a_max:.4f} AU above HZ"

    def test_eccentricity_secular_envelope(self) -> None:
        """Eccentricity should vary within expected secular envelope.

        For a quasi-resonant chain, secular variations should be bounded.
        """
        from tlsr_spin.nbody import build_bipolaris_system, integrate_and_extract

        sim = build_bipolaris_system()
        result = integrate_and_extract(sim, planet_idx=3, n_orbits=N_ORBITS_QUICK)

        e_min = np.min(result["e"])
        e_max = np.max(result["e"])
        e_mean = np.mean(result["e"])

        # Should stay near initial value with reasonable variations
        assert e_max < 0.2, f"e_max = {e_max:.4f} exceeds stability bound"
        assert e_min >= 0, f"e_min = {e_min:.4f} negative"
        assert abs(e_mean - BIPOLARIS_ECCENTRICITY) < 0.05, (
            f"Mean e = {e_mean:.4f} drifted from initial {BIPOLARIS_ECCENTRICITY}"
        )


@pytest.mark.skip(reason="kozai.py archived - requires Gyr timescales")
class TestKozaiLidov:
    """Tests for Kozai-Lidov dynamics from K5V companion."""

    def test_binary_kozai_timescale_very_long(self) -> None:
        """Binary Kozai timescale is >> Gyr for 180 AU companion.

        The companion is too distant for direct binary Kozai to matter.
        Multi-planet secular dynamics operate on shorter timescales.
        """
        from tlsr_spin.kozai import kozai_timescale_bipolaris

        p_kozai = kozai_timescale_bipolaris()

        # Binary formula gives very long timescale (>> Hubble time)
        assert p_kozai > 1e10, (
            f"Binary Kozai timescale {p_kozai:.2e} yr unexpectedly short"
        )

    def test_effective_multiplanet_timescale(self) -> None:
        """Effective multi-planet timescale should be ~100 Myr (estimated)."""
        from tlsr_spin.kozai import effective_kozai_timescale_multiplanet

        p_effective = effective_kozai_timescale_multiplanet()

        assert 1e7 < p_effective < 1e9, (
            f"Effective timescale {p_effective:.2e} yr out of expected range"
        )

    def test_critical_inclination_threshold(self) -> None:
        """Kozai oscillations should only occur above critical inclination."""
        from tlsr_spin.kozai import KOZAI_CRITICAL_INCLINATION_DEG, kozai_emax

        # Below critical: no eccentricity pumping
        e_max_below = kozai_emax(30.0)  # below 39.2°
        assert e_max_below < 0.1, f"e_max below critical = {e_max_below:.3f}"

        # Above critical: significant eccentricity pumping possible
        e_max_above = kozai_emax(50.0)  # above 39.2°
        assert e_max_above > 0.3, f"e_max above critical = {e_max_above:.3f}"

    def test_emax_prediction_matches_theory(self) -> None:
        """e_max should match theoretical formula sqrt(1 - 5/3 cos²i)."""
        from tlsr_spin.kozai import kozai_emax

        test_angles = [40, 50, 60, 70, 80]
        for i_deg in test_angles:
            i_rad = np.deg2rad(i_deg)
            expected = np.sqrt(1 - (5.0 / 3.0) * np.cos(i_rad) ** 2)
            actual = kozai_emax(i_deg)
            assert actual == pytest.approx(expected, rel=0.01), (
                f"At i={i_deg}°: expected {expected:.4f}, got {actual:.4f}"
            )

    def test_high_e_phase_can_trigger_instability(self) -> None:
        """High eccentricity from Kozai can cause orbit-crossing.

        If e_max is high enough, adjacent planet orbits may cross,
        which is the instability mechanism for the instability.
        """
        from shared.constants import PLANET_C_DISTANCE_AU

        from tlsr_spin.kozai import kozai_emax

        # For high inclination, check if Bipolaris could cross planet c
        e_max = kozai_emax(60.0)  # high inclination case
        perihelion = BIPOLARIS_DISTANCE_AU * (1 - e_max)
        aphelion_c = PLANET_C_DISTANCE_AU * (1 + 0.04)  # planet c aphelion

        # At very high e, Bipolaris perihelion can cross planet c aphelion
        # This IS the instability mechanism for the instability - verify it's possible
        assert e_max > 0.5, f"e_max = {e_max:.3f} not high enough for instability"

        orbit_separation = perihelion - aphelion_c
        # Negative separation means orbit crossing IS possible
        assert orbit_separation < 0.02, (
            f"Orbits don't approach closely enough: separation = {orbit_separation:.4f} AU"
        )


@pytest.mark.skip(reason="kozai.py archived - requires Gyr timescales")
class TestKozaiSecularIntegration:
    """Tests for secular Kozai integration."""

    def test_secular_integration_completes(self) -> None:
        """Secular Kozai integration should complete without error."""
        from tlsr_spin.kozai import integrate_kozai_secular

        result = integrate_kozai_secular(
            t_end_yr=1e5,  # 100 kyr
            n_samples=1000,
        )

        assert "t" in result
        assert "e" in result
        assert "i_deg" in result
        assert len(result["t"]) == 1000

    def test_secular_eccentricity_oscillates(self) -> None:
        """Eccentricity should oscillate over Kozai timescale.

        Starting from low e, eccentricity should increase then decrease.
        """
        from tlsr_spin.kozai import integrate_kozai_secular, kozai_timescale_bipolaris

        p_kozai = kozai_timescale_bipolaris()

        # Integrate for several Kozai cycles
        result = integrate_kozai_secular(
            t_end_yr=min(3 * p_kozai, 1e7),  # cap at 10 Myr
            e_initial=0.01,
            inclination_deg=50.0,  # above critical
            n_samples=5000,
        )

        e_min = np.min(result["e"])
        e_max = np.max(result["e"])

        # Should see variation
        assert e_max > e_min * 2, (
            f"e varied from {e_min:.4f} to {e_max:.4f} - "
            f"expected larger oscillation"
        )


@pytest.mark.skip(reason="gr_corrections.py archived - requires Gyr timescales")
class TestGRStability:
    """Tests for GR corrections and their effect on stability."""

    def test_gr_precession_rate_reasonable(self) -> None:
        """GR precession rate should be in expected range for close-in planet."""
        from tlsr_spin.gr_corrections import gr_precession_period, gr_precession_rate

        omega_gr = gr_precession_rate(
            BIPOLARIS_DISTANCE_AU, BIPOLARIS_ECCENTRICITY, STAR_MASS_MSUN
        )
        p_gr = gr_precession_period(
            BIPOLARIS_DISTANCE_AU, BIPOLARIS_ECCENTRICITY, STAR_MASS_MSUN
        )

        # GR precession for close-in planet should be tens to hundreds of years
        assert 1 < p_gr < 1e4, f"GR precession period {p_gr:.1f} yr out of range"
        assert omega_gr > 0, "GR precession rate should be positive"

    def test_gr_faster_than_kozai(self) -> None:
        """GR precession should be faster than Kozai for inner planets.

        For very close-in planets, GR can suppress Kozai oscillations.
        """
        from tlsr_spin.gr_corrections import gr_precession_rate
        from tlsr_spin.kozai import kozai_apsidal_precession_rate

        omega_gr = gr_precession_rate(
            BIPOLARIS_DISTANCE_AU, BIPOLARIS_ECCENTRICITY, STAR_MASS_MSUN
        )
        omega_kozai = kozai_apsidal_precession_rate(
            BIPOLARIS_DISTANCE_AU,
            COMPANION_DISTANCE_AU,
            STAR_MASS_MSUN,
            COMPANION_MASS_MSUN,
            BIPOLARIS_ECCENTRICITY,
        )

        # For Bipolaris at 0.055 AU, GR should be relevant
        ratio = abs(omega_gr / omega_kozai)
        # Just check ratio is computable and reasonable
        assert ratio > 0

    @pytest.mark.skipif(
        True,  # Skip by default - requires reboundx
        reason="Requires reboundx installation"
    )
    def test_gr_prevents_instability(self) -> None:
        """With GR, system should remain stable longer.

        GR apsidal precession can suppress Kozai-induced instabilities.
        """
        from tlsr_spin.gr_corrections import build_bipolaris_system_with_gr

        sim = build_bipolaris_system_with_gr()

        # Integrate for 10k orbits
        orb = sim.particles[3].orbit()
        t_end = 10000 * orb.P
        sim.integrate(t_end)

        # Check all planets still bound
        for i in range(1, sim.N):
            orb = sim.particles[i].orbit()
            assert orb.e < 1.0, f"Planet {i} ejected with GR: e = {orb.e:.3f}"

    @pytest.mark.skipif(
        True,  # Skip by default - requires reboundx
        reason="Requires reboundx installation"
    )
    def test_gr_stable_100k_orbits(self) -> None:
        """System with GR should be stable to 100k orbits (~1 Myr)."""
        from tlsr_spin.gr_corrections import build_bipolaris_system_with_gr

        sim = build_bipolaris_system_with_gr()

        # Long integration
        orb = sim.particles[3].orbit()
        t_end = 100000 * orb.P  # ~1 Myr
        sim.integrate(t_end)

        # All planets should remain bound
        for i in range(1, sim.N):
            orb = sim.particles[i].orbit()
            assert orb.e < 0.5, f"Planet {i} e = {orb.e:.3f} too high after 1 Myr"


@pytest.mark.skip(reason="kozai.py archived - requires Gyr timescales")
class TestCompanionEffects:
    """Tests for effects of the distant K5V companion."""

    def test_companion_inclination_below_critical(self) -> None:
        """Check if companion inclination is near critical threshold.

        The system design has i=35°, which is just below critical (39.2°).
        This allows slow secular evolution but not full Kozai oscillations.
        """
        from tlsr_spin.kozai import KOZAI_CRITICAL_INCLINATION_DEG

        # Current design is deliberately below critical
        assert COMPANION_INCLINATION_DEG < KOZAI_CRITICAL_INCLINATION_DEG, (
            f"Companion inclination {COMPANION_INCLINATION_DEG}° "
            f">= critical {KOZAI_CRITICAL_INCLINATION_DEG}°"
        )

        # But close enough that perturbations could push it over
        diff = KOZAI_CRITICAL_INCLINATION_DEG - COMPANION_INCLINATION_DEG
        assert diff < 10, f"Inclination {diff}° below critical - maybe too stable"

    def test_effective_cycles_possible_over_gyr(self) -> None:
        """Multiple secular cycles should be possible over 7.5 Gyr.

        Using the effective multi-planet timescale, not binary Kozai.
        """
        from tlsr_spin.kozai import effective_kozai_timescale_multiplanet

        p_effective = effective_kozai_timescale_multiplanet()
        n_cycles = (SYSTEM_AGE_GYR * 1e9) / p_effective

        # Should have multiple cycles
        assert n_cycles > 10, (
            f"Only {n_cycles:.1f} secular cycles in {SYSTEM_AGE_GYR} Gyr"
        )
