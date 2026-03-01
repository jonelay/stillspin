"""Shared fixtures for chain survey tests."""

import pytest

from chain_survey.chain_types import (
    ChainConfig,
    OrbitalResult,
    PerturbationProbe,
    PlanetSpec,
    SpinSurveyResult,
    SystemArchitecture,
)


@pytest.fixture
def known_4planet_config():
    """A ChainConfig matching a Bipolaris-like 4-planet system."""
    return ChainConfig(
        stellar_mass_msun=0.15,
        n_planets=4,
        anchor_idx=2,
        seed=42,
        mmr_sequence=("3:2", "4:3", "3:2"),
        planet_masses_mearth=(0.8, 1.2, 1.0, 0.6),
    )


@pytest.fixture
def sample_planets():
    """Four planets in a plausible configuration."""
    return (
        PlanetSpec(
            mass_mearth=0.8, radius_rearth=0.94, period_days=10.0,
            distance_au=0.045, eccentricity=0.02, mmr_label=None,
        ),
        PlanetSpec(
            mass_mearth=1.2, radius_rearth=1.05, period_days=15.0,
            distance_au=0.058, eccentricity=0.03, mmr_label="3:2",
        ),
        PlanetSpec(
            mass_mearth=1.0, radius_rearth=1.0, period_days=20.0,
            distance_au=0.070, eccentricity=0.02, mmr_label="4:3",
        ),
        PlanetSpec(
            mass_mearth=0.6, radius_rearth=0.87, period_days=30.0,
            distance_au=0.090, eccentricity=0.04, mmr_label="3:2",
        ),
    )


@pytest.fixture
def known_system(known_4planet_config, sample_planets):
    """A SystemArchitecture from known config."""
    return SystemArchitecture(
        config=known_4planet_config,
        stellar_mass_msun=0.15,
        stellar_luminosity_lsun=0.15**4,
        planets=sample_planets,
        hz_inner_au=0.048,
        hz_outer_au=0.098,
        hz_planet_indices=(2, 3),
        mutual_hill_separations=(10.0, 12.0, 15.0),
        formation_stable=True,
    )


@pytest.fixture
def mock_orbital_result(known_system):
    """An OrbitalResult with PARTIAL_BREAK status."""
    return OrbitalResult(
        system_id=known_system.system_id,
        status="PARTIAL_BREAK",
        final_planets=[
            {"idx": 1, "a_au": 0.045, "e": 0.025, "incl_deg": 0.1, "survived": True},
            {"idx": 2, "a_au": 0.058, "e": 0.035, "incl_deg": 0.2, "survived": True},
            {"idx": 3, "a_au": 0.071, "e": 0.022, "incl_deg": 0.1, "survived": True},
            {"idx": 4, "a_au": 0.091, "e": 0.045, "incl_deg": 0.3, "survived": True},
        ],
        breaking_events=[{
            "time_myr": 120.5,
            "type": "mmr_break",
            "pair_idx": 1,
            "initial_ratio": 1.333,
            "current_ratio": 1.402,
        }],
        n_survivors=4,
        hz_planet_survived=True,
        hz_planet_idx=3,
        elapsed_s=100.0,
    )


@pytest.fixture
def mock_probe_pass(mock_orbital_result):
    """A PerturbationProbe with PASS verdict."""
    return PerturbationProbe(
        system_id=mock_orbital_result.system_id,
        hz_planet_idx=3,
        rms_dn_over_n=0.005,
        rms_de=0.003,
        max_e=0.08,
        mean_n_rad_s=3.8e-6,
        dominant_periods_yr=[15.2, 8.1, 3.5, 22.0, 45.0],
        filter_verdict="PASS",
        elapsed_s=10.0,
    )


@pytest.fixture
def mock_probe_reject(mock_orbital_result):
    """A PerturbationProbe with REJECT_WEAK verdict."""
    return PerturbationProbe(
        system_id=mock_orbital_result.system_id,
        hz_planet_idx=3,
        rms_dn_over_n=0.0001,
        rms_de=0.0005,
        max_e=0.01,
        mean_n_rad_s=3.8e-6,
        dominant_periods_yr=[15.2, 8.1],
        filter_verdict="REJECT_WEAK",
        elapsed_s=10.0,
    )


@pytest.fixture
def quick_params():
    """Quick mode params for fast tests."""
    return {"t_end_myr": 1.0, "probe_years": 500, "spin_years": 500}
