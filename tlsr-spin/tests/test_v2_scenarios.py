"""Tests for v2 scenario definitions and overrides.

Verifies that scenarios A-G are correctly defined and have
physically consistent override values.
"""

import pytest

from shared.constants import (
    BIPOLARIS_ECCENTRICITY,
    BIPOLARIS_TIDAL_Q,
    BIPOLARIS_TRIAXIALITY,
    PLANET_B_ECCENTRICITY,
    PLANET_C_ECCENTRICITY,
    PLANET_D_ECCENTRICITY,
    STAR_MASS_MSUN,
)
from shared.scenarios import SCENARIOS, get_scenario


class TestScenarioOverrides:
    """Tests for scenario override correctness."""

    def test_scenario_a_baseline(self) -> None:
        """Scenario A (baseline) should have no overrides."""
        overrides = get_scenario("A")
        assert overrides == {}, "Scenario A should have empty overrides"

    def test_scenario_b_lower_eccentricities(self) -> None:
        """Scenario B should have lower eccentricities (pre-Breaking)."""
        overrides = get_scenario("B")

        # Check all eccentricities are overridden to 0.01
        assert overrides["PLANET_B_ECCENTRICITY"] == 0.01
        assert overrides["PLANET_C_ECCENTRICITY"] == 0.01
        assert overrides["BIPOLARIS_ECCENTRICITY"] == 0.01
        assert overrides["PLANET_D_ECCENTRICITY"] == 0.01

        # Planet e should not be ejected
        assert overrides["PLANET_E_EJECTED"] is False

    def test_scenario_c_high_q(self) -> None:
        """Scenario C should override Q to 100."""
        overrides = get_scenario("C")
        assert overrides.get("BIPOLARIS_TIDAL_Q") == 100

    def test_scenario_d_low_triaxiality(self) -> None:
        """Scenario D should override triaxiality to 1e-5."""
        overrides = get_scenario("D")
        assert overrides.get("BIPOLARIS_TRIAXIALITY") == pytest.approx(1e-5)

    def test_scenario_e_high_triaxiality(self) -> None:
        """Scenario E should override triaxiality to 1e-3."""
        overrides = get_scenario("E")
        assert overrides.get("BIPOLARIS_TRIAXIALITY") == pytest.approx(1e-3)

    def test_scenario_f_trappist1_validation(self) -> None:
        """Scenario F should have TRAPPIST-1 stellar parameters."""
        overrides = get_scenario("F")

        # Stellar mass should be TRAPPIST-1
        assert overrides.get("STAR_MASS_MSUN") == pytest.approx(0.0898)

        # Should have planetary parameters for TRAPPIST-1 f
        assert "BIPOLARIS_DISTANCE_AU" in overrides
        assert overrides["BIPOLARIS_DISTANCE_AU"] == pytest.approx(0.03853, rel=0.01)

    def test_all_scenarios_have_labels(self) -> None:
        """All scenarios should have a label and description."""
        for name, scenario in SCENARIOS.items():
            assert "label" in scenario, f"Scenario {name} missing label"
            assert "description" in scenario, f"Scenario {name} missing description"
            assert len(scenario["label"]) > 0
            assert len(scenario["description"]) > 0


class TestScenarioPhysicalConsistency:
    """Tests for physical consistency between scenarios."""

    def test_a_eccentricities_higher_than_b(self) -> None:
        """Scenario A (post-Breaking) should have higher eccentricities than B."""
        overrides_a = get_scenario("A")
        overrides_b = get_scenario("B")

        # A uses defaults, B has lower values
        e_a = BIPOLARIS_ECCENTRICITY
        e_b = overrides_b.get("BIPOLARIS_ECCENTRICITY", BIPOLARIS_ECCENTRICITY)

        assert e_a > e_b, (
            f"Scenario A eccentricity {e_a} should be > Scenario B {e_b}"
        )

    def test_scenario_d_triax_lower_than_baseline(self) -> None:
        """Scenario D triaxiality should be lower than baseline."""
        overrides_d = get_scenario("D")
        triax_d = overrides_d.get("BIPOLARIS_TRIAXIALITY", BIPOLARIS_TRIAXIALITY)

        assert triax_d < BIPOLARIS_TRIAXIALITY

    def test_scenario_e_triax_higher_than_baseline(self) -> None:
        """Scenario E triaxiality should be higher than baseline."""
        overrides_e = get_scenario("E")
        triax_e = overrides_e.get("BIPOLARIS_TRIAXIALITY", BIPOLARIS_TRIAXIALITY)

        assert triax_e > BIPOLARIS_TRIAXIALITY

    def test_scenario_c_q_higher_than_baseline(self) -> None:
        """Scenario C tidal Q should be higher than baseline."""
        overrides_c = get_scenario("C")
        q_c = overrides_c.get("BIPOLARIS_TIDAL_Q", BIPOLARIS_TIDAL_Q)

        assert q_c > BIPOLARIS_TIDAL_Q

    def test_f_star_mass_lower_than_bipolaris(self) -> None:
        """TRAPPIST-1 (Scenario F) stellar mass should be lower than primary."""
        overrides_f = get_scenario("F")
        m_f = overrides_f.get("STAR_MASS_MSUN", STAR_MASS_MSUN)

        # TRAPPIST-1 is 0.09 M_sun, primary is 0.15 M_sun
        assert m_f < STAR_MASS_MSUN

    def test_invalid_scenario_raises(self) -> None:
        """Getting an invalid scenario should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown scenario"):
            get_scenario("Z")


class TestScenarioDerivation:
    """Tests for derived values with scenario overrides."""

    def test_scenario_b_all_planets_lower_e(self) -> None:
        """In Scenario B, all 4 remaining planets should have e=0.01."""
        overrides = get_scenario("B")

        for key in [
            "PLANET_B_ECCENTRICITY",
            "PLANET_C_ECCENTRICITY",
            "BIPOLARIS_ECCENTRICITY",
            "PLANET_D_ECCENTRICITY",
        ]:
            assert key in overrides
            assert overrides[key] == 0.01

    def test_scenario_g_sweepable_distance(self) -> None:
        """Scenario G should have TRAPPIST-1 star but no distance override.

        Distance is meant to be set via CLI --a-override.
        """
        overrides = get_scenario("G")

        # Should have stellar parameters
        assert "STAR_MASS_MSUN" in overrides

        # Should NOT have distance override (it's CLI-sweepable)
        assert "BIPOLARIS_DISTANCE_AU" not in overrides

    def test_scenario_f_complete_planet_params(self) -> None:
        """Scenario F should have complete TRAPPIST-1 f parameters."""
        overrides = get_scenario("F")

        required_keys = [
            "STAR_MASS_MSUN",
            "STAR_MASS",
            "STAR_LUMINOSITY_LSUN",
            "STAR_LUMINOSITY",
            "BIPOLARIS_DISTANCE_AU",
            "BIPOLARIS_MASS",
            "BIPOLARIS_RADIUS",
        ]

        for key in required_keys:
            assert key in overrides, f"Scenario F missing {key}"
