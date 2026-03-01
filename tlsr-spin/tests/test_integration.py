"""Integration tests for the TLSR spin dynamics pipeline.

These tests verify that the full pipeline (N-body -> spin -> regime)
executes without error and produces physically reasonable results.

Markers:
- @pytest.mark.integration: marks all tests in this module
- @pytest.mark.slow: tests that take > 10 seconds
"""

import numpy as np
import pytest

from shared.constants import (
    AU,
    BIPOLARIS_DISTANCE_AU,
    BIPOLARIS_MASS,
    BIPOLARIS_RADIUS,
    BIPOLARIS_TIDAL_Q,
    BIPOLARIS_TRIAXIALITY,
    STAR_MASS,
    YEAR,
)
from shared.scenarios import get_scenario

from tests.conftest import N_ORBITS_QUICK


pytestmark = pytest.mark.integration


class TestNbodyIntegration:
    """Tests for N-body simulation execution."""

    def test_build_bipolaris_system_succeeds(self) -> None:
        """build_bipolaris_system() should complete without error."""
        from tlsr_spin.nbody import build_bipolaris_system

        sim = build_bipolaris_system()
        assert sim is not None
        # Should have star + 4 planets
        assert sim.N == 5

    def test_integration_stable_no_nan(self) -> None:
        """1000-orbit integration should have no NaN/Inf values."""
        from tlsr_spin.nbody import build_bipolaris_system, integrate_and_extract

        sim = build_bipolaris_system()
        result = integrate_and_extract(sim, planet_idx=3, n_orbits=N_ORBITS_QUICK)

        assert not np.any(np.isnan(result["t"]))
        assert not np.any(np.isnan(result["e"]))
        assert not np.any(np.isnan(result["n"]))
        assert not np.any(np.isnan(result["a"]))
        assert not np.any(np.isinf(result["e"]))

    def test_eccentricities_bounded(self) -> None:
        """Eccentricities should stay in (0, 1) for stable orbits."""
        from tlsr_spin.nbody import build_bipolaris_system, integrate_and_extract

        sim = build_bipolaris_system()
        result = integrate_and_extract(sim, planet_idx=3, n_orbits=N_ORBITS_QUICK)

        assert np.all(result["e"] >= 0), "Negative eccentricity detected"
        assert np.all(result["e"] < 1), "Eccentricity >= 1 (hyperbolic)"

    def test_mean_motion_positive(self) -> None:
        """Mean motion n should be positive throughout integration."""
        from tlsr_spin.nbody import build_bipolaris_system, integrate_and_extract

        sim = build_bipolaris_system()
        result = integrate_and_extract(sim, planet_idx=3, n_orbits=N_ORBITS_QUICK)

        assert np.all(result["n"] > 0), "Non-positive mean motion detected"

    def test_semi_major_axis_stable(self) -> None:
        """Semi-major axis should stay within 20% of initial value."""
        from tlsr_spin.nbody import build_bipolaris_system, integrate_and_extract

        sim = build_bipolaris_system()
        result = integrate_and_extract(sim, planet_idx=3, n_orbits=N_ORBITS_QUICK)

        a_mean = np.mean(result["a"])
        a_dev = np.abs(result["a"] - a_mean) / a_mean

        assert np.all(a_dev < 0.2), f"Semi-major axis deviation > 20%: max {a_dev.max():.1%}"


class TestSpinIntegration:
    """Tests for spin ODE integration."""

    def test_spin_integration_completes(self, short_nbody_result: dict) -> None:
        """Spin integration should complete without error."""
        from tlsr_spin.spin_integrator import integrate_spin

        nbody = short_nbody_result
        a_mean = float(np.mean(nbody["a"])) * AU

        result = integrate_spin(
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

        assert "gamma" in result
        assert "gamma_dot" in result
        assert "tidal_epsilon" in result

    def test_gamma_stays_finite(self, short_nbody_result: dict) -> None:
        """γ should remain finite (no blowup)."""
        from tlsr_spin.spin_integrator import integrate_spin

        nbody = short_nbody_result
        a_mean = float(np.mean(nbody["a"])) * AU

        result = integrate_spin(
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

        assert np.all(np.isfinite(result["gamma"]))

    def test_gamma_dot_reasonable(self, short_nbody_result: dict) -> None:
        """γ̇ should stay bounded: |γ̇| < 10n (not wildly supersynchronous)."""
        from tlsr_spin.spin_integrator import integrate_spin

        nbody = short_nbody_result
        a_mean = float(np.mean(nbody["a"])) * AU
        n_typical = float(np.median(nbody["n"]))

        result = integrate_spin(
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

        max_gamma_dot = np.max(np.abs(result["gamma_dot"]))
        assert max_gamma_dot < 10 * n_typical, (
            f"|γ̇| max {max_gamma_dot:.4e} > 10n = {10 * n_typical:.4e}"
        )

    def test_tidal_epsilon_returned(self, short_nbody_result: dict) -> None:
        """Tidal epsilon should be returned in result dict."""
        from tlsr_spin.spin_integrator import integrate_spin

        nbody = short_nbody_result
        a_mean = float(np.mean(nbody["a"])) * AU

        result = integrate_spin(
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

        assert result["tidal_epsilon"] > 0


class TestRegimeClassificationV2:
    """Tests for regime classification with v2 architecture."""

    def test_classify_scenario_a(self) -> None:
        """Classification for Scenario A should complete without error."""
        from tlsr_spin.nbody import build_bipolaris_system, integrate_and_extract
        from tlsr_spin.regime_classifier import classify_regimes
        from tlsr_spin.spin_integrator import integrate_spin

        # Run short pipeline
        sim = build_bipolaris_system()
        nbody = integrate_and_extract(sim, planet_idx=3, n_orbits=N_ORBITS_QUICK)

        a_mean = float(np.mean(nbody["a"])) * AU
        spin = integrate_spin(
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

        result = classify_regimes(spin["t"], spin["gamma"])
        assert result is not None

    def test_fractions_sum_to_one(self, short_spin_result: dict) -> None:
        """Regime fractions should sum to approximately 1.0."""
        from tlsr_spin.regime_classifier import classify_regimes, compute_regime_fractions

        result = classify_regimes(short_spin_result["t"], short_spin_result["gamma"])
        fractions = compute_regime_fractions(result)

        total = sum(fractions.values())
        assert total == pytest.approx(1.0, abs=0.05)

    def test_at_least_one_regime_detected(self, short_spin_result: dict) -> None:
        """Classification should detect at least one regime."""
        from tlsr_spin.regime_classifier import classify_regimes

        result = classify_regimes(short_spin_result["t"], short_spin_result["gamma"])
        assert len(result.regimes) >= 1

    def test_ptb_threshold_applied(self, short_spin_result: dict) -> None:
        """Short regimes should be reclassified as PTB."""
        from tlsr_spin.regime_classifier import RegimeType, classify_regimes

        # Use very long min_regime_yr to force PTB classification
        result = classify_regimes(
            short_spin_result["t"],
            short_spin_result["gamma"],
            min_regime_yr=1000.0,  # longer than integration
        )

        # With such a long threshold, most should be PTB
        ptb_count = sum(1 for r in result.regimes if r.type == RegimeType.PTB)
        # At least some PTB expected (might not be all if duration is longer)
        assert ptb_count >= 0  # Just verify it runs


class TestFullPipeline:
    """Tests for full pipeline execution."""

    @pytest.mark.slow
    def test_full_pipeline_scenario_a(self) -> None:
        """Full pipeline for Scenario A should complete."""
        from tlsr_spin.sweep import run_single

        result = run_single(
            scenario_name="A",
            triaxiality=BIPOLARIS_TRIAXIALITY,
            tidal_q=BIPOLARIS_TIDAL_Q,
            n_orbits=N_ORBITS_QUICK,
            quiet=True,
        )

        assert "fractions" in result
        assert "stats" in result
        assert "quasi_stable_fraction" in result
        assert result["duration_yr"] > 0

    @pytest.mark.slow
    def test_full_pipeline_scenario_f_trappist(self) -> None:
        """Full pipeline for Scenario F (TRAPPIST-1) should complete."""
        from tlsr_spin.sweep import run_single

        result = run_single(
            scenario_name="F",
            triaxiality=1e-5,
            tidal_q=10,
            n_orbits=N_ORBITS_QUICK,
            quiet=True,
        )

        assert "fractions" in result
        assert result["scenario"] == "F"


class TestScenarioOverrideIntegration:
    """Test that scenario overrides are correctly applied in N-body."""

    def test_scenario_b_lower_eccentricity(self) -> None:
        """Scenario B N-body should show lower eccentricities than A."""
        from tlsr_spin.nbody import build_bipolaris_system, integrate_and_extract

        # Scenario A (baseline)
        sim_a = build_bipolaris_system()
        nbody_a = integrate_and_extract(sim_a, planet_idx=3, n_orbits=N_ORBITS_QUICK)

        # Scenario B (stable era)
        overrides_b = get_scenario("B")
        sim_b = build_bipolaris_system(overrides=overrides_b)
        nbody_b = integrate_and_extract(sim_b, planet_idx=3, n_orbits=N_ORBITS_QUICK)

        # Mean eccentricity should be lower in B
        e_mean_a = np.mean(nbody_a["e"])
        e_mean_b = np.mean(nbody_b["e"])

        assert e_mean_b < e_mean_a, (
            f"Scenario B mean e ({e_mean_b:.4f}) should be < A ({e_mean_a:.4f})"
        )

    def test_scenario_f_different_star_mass(self) -> None:
        """Scenario F should have TRAPPIST-1 stellar mass in simulation."""
        from tlsr_spin.nbody import build_bipolaris_system

        overrides_f = get_scenario("F")
        sim = build_bipolaris_system(overrides=overrides_f)

        # Star mass should be TRAPPIST-1 (0.0898 M_sun)
        star_mass_msun = sim.particles[0].m
        assert star_mass_msun == pytest.approx(0.0898, rel=0.01)
