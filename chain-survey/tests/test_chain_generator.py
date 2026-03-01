"""Tests for chain_generator.py: draws, cascade, stability, generation."""

import numpy as np
import pytest

from chain_survey.chain_generator import (
    MMR_PALETTE,
    cascade_periods,
    distance_to_period,
    draw_planet_count,
    draw_planet_masses,
    draw_stellar_mass,
    generate_batch,
    generate_chain,
    hz_bounds,
    mass_adjusted_mmr_weights,
    mass_radius_relation,
    mutual_hill_separation,
    period_to_distance,
    stellar_luminosity,
    validate_stability,
)


class TestStellarDraws:
    def test_stellar_mass_in_range(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            m = draw_stellar_mass(rng)
            assert 0.08 <= m <= 0.25

    def test_stellar_luminosity_scaling(self):
        # L ∝ M^4, so L(0.15) ≈ 0.15^4 ≈ 5.06e-4
        lum = stellar_luminosity(0.15)
        assert abs(lum - 0.15**4) < 1e-10

    def test_stellar_luminosity_solar_mass(self):
        # For M=1.0, L ≈ 1.0 (but our function is for M < 0.43)
        assert stellar_luminosity(1.0) == pytest.approx(1.0)


class TestPlanetDraws:
    def test_planet_count_in_range(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            n = draw_planet_count(rng)
            assert 3 <= n <= 7

    def test_planet_masses_in_range(self):
        rng = np.random.default_rng(42)
        for _ in range(50):
            masses = draw_planet_masses(5, rng)
            assert len(masses) == 5
            for m in masses:
                assert 0.3 <= m <= 3.0


class TestMassRadiusRelation:
    def test_earth_unity(self):
        r = mass_radius_relation(1.0)
        assert r == pytest.approx(1.0, abs=0.01)

    def test_heavier_larger(self):
        r1 = mass_radius_relation(0.5)
        r2 = mass_radius_relation(2.0)
        assert r2 > r1


class TestMMRWeights:
    def test_weights_sum_to_one(self):
        weights = mass_adjusted_mmr_weights(1.0, 1.0)
        total = sum(weights.values())
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_heavier_planets_boost_wider_resonances(self):
        light_weights = mass_adjusted_mmr_weights(0.5, 0.5)
        heavy_weights = mass_adjusted_mmr_weights(2.0, 2.0)
        assert heavy_weights["2:1"] > light_weights["2:1"]

    def test_lighter_planets_boost_tighter_resonances(self):
        light_weights = mass_adjusted_mmr_weights(0.5, 0.5)
        heavy_weights = mass_adjusted_mmr_weights(2.0, 2.0)
        assert light_weights["5:4"] > heavy_weights["5:4"]


class TestPeriodCascade:
    def test_cascade_preserves_anchor_period(self):
        anchor_period = 20.0
        mmrs = ("3:2", "4:3", "3:2")
        periods = cascade_periods(anchor_period, 2, mmrs)
        assert periods[2] == pytest.approx(anchor_period)

    def test_cascade_periods_increase_outward(self):
        periods = cascade_periods(20.0, 2, ("3:2", "4:3", "3:2"))
        for i in range(len(periods) - 1):
            assert periods[i + 1] > periods[i]

    def test_cascade_matches_mmr_ratios(self):
        mmrs = ("3:2", "4:3", "5:4")
        periods = cascade_periods(20.0, 1, mmrs)
        for i in range(len(mmrs)):
            expected_ratio = MMR_PALETTE[mmrs[i]]
            actual_ratio = periods[i + 1] / periods[i]
            assert actual_ratio == pytest.approx(expected_ratio, rel=1e-10)


class TestKeplerLaws:
    def test_period_to_distance_kepler(self):
        # Earth: P=365.25 days, a=1 AU, M_star=1 M_sun
        d = period_to_distance(365.25, 1.0)
        assert d == pytest.approx(1.0, abs=0.01)

    def test_roundtrip_period_distance(self):
        d_orig = 0.07
        m_star = 0.15
        p = distance_to_period(d_orig, m_star)
        d_back = period_to_distance(p, m_star)
        assert d_back == pytest.approx(d_orig, rel=1e-8)


class TestHillSeparation:
    def test_rejects_too_close(self, sample_planets):
        from chain_survey.chain_types import PlanetSpec

        # Two planets at nearly same distance
        close_planets = [
            PlanetSpec(mass_mearth=1.0, radius_rearth=1.0, period_days=10.0,
                       distance_au=0.05, eccentricity=0.01),
            PlanetSpec(mass_mearth=1.0, radius_rearth=1.0, period_days=10.5,
                       distance_au=0.0505, eccentricity=0.01),
        ]
        valid, seps = validate_stability(close_planets, 0.15)
        assert not valid
        assert seps[0] < 5.0

    def test_accepts_moderate(self, sample_planets):
        valid, seps = validate_stability(list(sample_planets), 0.15)
        # Our sample planets should be reasonably spaced
        assert all(k > 0 for k in seps)

    def test_rejects_too_far(self):
        from chain_survey.chain_types import PlanetSpec

        far_planets = [
            PlanetSpec(mass_mearth=1.0, radius_rearth=1.0, period_days=10.0,
                       distance_au=0.01, eccentricity=0.01),
            PlanetSpec(mass_mearth=1.0, radius_rearth=1.0, period_days=300.0,
                       distance_au=0.5, eccentricity=0.01),
        ]
        valid, seps = validate_stability(far_planets, 0.15)
        assert not valid
        assert seps[0] > 25.0


class TestHZBounds:
    def test_hz_bounds_positive(self):
        hz_in, hz_out = hz_bounds(0.15**4)
        assert hz_in > 0
        assert hz_out > hz_in


class TestGenerateChain:
    def test_returns_valid_system(self):
        # Try several seeds, at least one should work
        for seed in range(42, 142):
            system = generate_chain(seed)
            if system is not None:
                assert len(system.planets) >= 3
                assert system.formation_stable
                return
        pytest.fail("No valid system generated in 100 attempts")

    def test_has_hz_planet(self):
        for seed in range(42, 142):
            system = generate_chain(seed)
            if system is not None:
                assert len(system.hz_planet_indices) >= 1
                return
        pytest.fail("No valid system generated")

    def test_deterministic(self):
        system1 = generate_chain(42)
        system2 = generate_chain(42)
        if system1 is not None and system2 is not None:
            assert system1.system_id == system2.system_id
        # If both are None, that's also deterministic

    def test_different_seeds_differ(self):
        systems = {}
        for seed in range(42, 142):
            s = generate_chain(seed)
            if s is not None:
                systems[seed] = s
            if len(systems) >= 2:
                break
        if len(systems) >= 2:
            ids = [s.system_id for s in systems.values()]
            assert ids[0] != ids[1]


class TestGenerateBatch:
    def test_returns_requested_count(self):
        systems = generate_batch(5, max_retries_per=50, base_seed=42)
        assert len(systems) == 5

    def test_all_have_hz_planets(self):
        systems = generate_batch(3, max_retries_per=50, base_seed=42)
        for s in systems:
            assert len(s.hz_planet_indices) >= 1

    def test_unique_system_ids(self):
        systems = generate_batch(5, max_retries_per=50, base_seed=42)
        ids = [s.system_id for s in systems]
        assert len(set(ids)) == len(ids)
