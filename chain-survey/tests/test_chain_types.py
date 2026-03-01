"""Tests for chain_types.py dataclasses: validation, serialization, config_id."""

import pytest

from chain_survey.chain_types import (
    ChainConfig,
    OrbitalResult,
    PerturbationProbe,
    PlanetSpec,
    SpinSurveyResult,
    SystemArchitecture,
)


class TestPlanetSpec:
    def test_valid_planet(self):
        p = PlanetSpec(
            mass_mearth=1.0, radius_rearth=1.0, period_days=19.2,
            distance_au=0.07, eccentricity=0.05,
        )
        assert p.mass_mearth == 1.0

    def test_rejects_mass_below_01(self):
        with pytest.raises(ValueError, match="mass="):
            PlanetSpec(
                mass_mearth=0.05, radius_rearth=0.5, period_days=10.0,
                distance_au=0.05, eccentricity=0.01,
            )

    def test_rejects_distance_above_1(self):
        with pytest.raises(ValueError, match="distance="):
            PlanetSpec(
                mass_mearth=1.0, radius_rearth=1.0, period_days=100.0,
                distance_au=1.5, eccentricity=0.01,
            )

    def test_rejects_eccentricity_above_03(self):
        with pytest.raises(ValueError, match="eccentricity="):
            PlanetSpec(
                mass_mearth=1.0, radius_rearth=1.0, period_days=10.0,
                distance_au=0.05, eccentricity=0.5,
            )

    def test_to_dict_roundtrip(self):
        p = PlanetSpec(
            mass_mearth=1.0, radius_rearth=1.0, period_days=19.2,
            distance_au=0.07, eccentricity=0.05, mmr_label="3:2",
        )
        assert PlanetSpec.from_dict(p.to_dict()) == p


class TestChainConfig:
    def test_valid_config(self, known_4planet_config):
        assert known_4planet_config.n_planets == 4
        assert len(known_4planet_config.config_id) == 12

    def test_rejects_stellar_mass_below_008(self):
        with pytest.raises(ValueError, match="stellar_mass="):
            ChainConfig(
                stellar_mass_msun=0.05, n_planets=3, anchor_idx=1, seed=1,
                mmr_sequence=("3:2", "4:3"),
                planet_masses_mearth=(1.0, 1.0, 1.0),
            )

    def test_rejects_stellar_mass_above_025(self):
        with pytest.raises(ValueError, match="stellar_mass="):
            ChainConfig(
                stellar_mass_msun=0.30, n_planets=3, anchor_idx=1, seed=1,
                mmr_sequence=("3:2", "4:3"),
                planet_masses_mearth=(1.0, 1.0, 1.0),
            )

    def test_rejects_n_planets_below_3(self):
        with pytest.raises(ValueError, match="n_planets="):
            ChainConfig(
                stellar_mass_msun=0.15, n_planets=2, anchor_idx=0, seed=1,
                mmr_sequence=("3:2",),
                planet_masses_mearth=(1.0, 1.0),
            )

    def test_rejects_n_planets_above_7(self):
        with pytest.raises(ValueError, match="n_planets="):
            ChainConfig(
                stellar_mass_msun=0.15, n_planets=8, anchor_idx=3, seed=1,
                mmr_sequence=("3:2",) * 7,
                planet_masses_mearth=(1.0,) * 8,
            )

    def test_rejects_anchor_out_of_range(self):
        with pytest.raises(ValueError, match="anchor_idx="):
            ChainConfig(
                stellar_mass_msun=0.15, n_planets=3, anchor_idx=5, seed=1,
                mmr_sequence=("3:2", "4:3"),
                planet_masses_mearth=(1.0, 1.0, 1.0),
            )

    def test_rejects_wrong_mmr_length(self):
        with pytest.raises(ValueError, match="mmr_sequence length"):
            ChainConfig(
                stellar_mass_msun=0.15, n_planets=3, anchor_idx=1, seed=1,
                mmr_sequence=("3:2",),
                planet_masses_mearth=(1.0, 1.0, 1.0),
            )

    def test_rejects_wrong_mass_length(self):
        with pytest.raises(ValueError, match="planet_masses length"):
            ChainConfig(
                stellar_mass_msun=0.15, n_planets=3, anchor_idx=1, seed=1,
                mmr_sequence=("3:2", "4:3"),
                planet_masses_mearth=(1.0, 1.0),
            )

    def test_config_id_deterministic(self):
        c1 = ChainConfig(
            stellar_mass_msun=0.15, n_planets=3, anchor_idx=1, seed=42,
            mmr_sequence=("3:2", "4:3"),
            planet_masses_mearth=(1.0, 1.0, 1.0),
        )
        c2 = ChainConfig(
            stellar_mass_msun=0.15, n_planets=3, anchor_idx=1, seed=42,
            mmr_sequence=("3:2", "4:3"),
            planet_masses_mearth=(1.0, 1.0, 1.0),
        )
        assert c1.config_id == c2.config_id

    def test_config_id_unique_for_different_seeds(self):
        c1 = ChainConfig(
            stellar_mass_msun=0.15, n_planets=3, anchor_idx=1, seed=42,
            mmr_sequence=("3:2", "4:3"),
            planet_masses_mearth=(1.0, 1.0, 1.0),
        )
        c2 = ChainConfig(
            stellar_mass_msun=0.15, n_planets=3, anchor_idx=1, seed=43,
            mmr_sequence=("3:2", "4:3"),
            planet_masses_mearth=(1.0, 1.0, 1.0),
        )
        assert c1.config_id != c2.config_id

    def test_to_dict_roundtrip(self, known_4planet_config):
        d = known_4planet_config.to_dict()
        restored = ChainConfig.from_dict(d)
        assert restored == known_4planet_config
        assert restored.config_id == known_4planet_config.config_id


class TestSystemArchitecture:
    def test_system_id_matches_config(self, known_system):
        assert known_system.system_id == known_system.config.config_id

    def test_to_dict_roundtrip(self, known_system):
        d = known_system.to_dict()
        restored = SystemArchitecture.from_dict(d)
        assert restored.system_id == known_system.system_id
        assert len(restored.planets) == len(known_system.planets)
        assert restored.formation_stable == known_system.formation_stable


class TestOrbitalResult:
    def test_to_dict_roundtrip(self, mock_orbital_result):
        d = mock_orbital_result.to_dict()
        restored = OrbitalResult.from_dict(d)
        assert restored.system_id == mock_orbital_result.system_id
        assert restored.status == "PARTIAL_BREAK"
        assert restored.n_survivors == 4


class TestPerturbationProbe:
    def test_to_dict_roundtrip(self, mock_probe_pass):
        d = mock_probe_pass.to_dict()
        restored = PerturbationProbe.from_dict(d)
        assert restored.system_id == mock_probe_pass.system_id
        assert restored.filter_verdict == "PASS"
        assert len(restored.dominant_periods_yr) == 5


class TestSpinSurveyResult:
    def test_to_dict_roundtrip(self):
        r = SpinSurveyResult(
            system_id="abc123def456",
            tidal_q=22,
            triaxiality=3e-5,
            fractions={"TL_ZERO": 0.1, "TL_PI": 0.5, "SPINNING": 0.0, "PTB": 0.4},
            is_flipflop=False,
            episodes=[{"type": "TL_PI", "duration_yr": 50.0}],
            status="OK",
            elapsed_s=30.0,
        )
        d = r.to_dict()
        restored = SpinSurveyResult.from_dict(d)
        assert restored.system_id == r.system_id
        assert restored.tidal_q == 22
        assert restored.is_flipflop is False
