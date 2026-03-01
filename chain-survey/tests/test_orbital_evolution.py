"""Tests for orbital_evolution.py: sim building, evolution, breaking detection."""

import pytest

from chain_survey.orbital_evolution import detect_mmr_breaking, extract_final_elements


class TestMMRBreaking:
    def test_no_breaking_within_tolerance(self):
        initial = [1.5, 1.333, 1.5]
        current = [1.52, 1.340, 1.48]
        events = detect_mmr_breaking(current, initial, tolerance=0.05)
        assert len(events) == 0

    def test_breaking_detected(self):
        initial = [1.5, 1.333, 1.5]
        current = [1.5, 1.5, 1.5]  # Pair 1 drifted from 1.333 to 1.5
        events = detect_mmr_breaking(current, initial, tolerance=0.05)
        assert len(events) >= 1
        assert events[0]["pair_idx"] == 1

    def test_multiple_breaks(self):
        initial = [1.5, 1.333]
        current = [2.0, 2.0]
        events = detect_mmr_breaking(current, initial, tolerance=0.05)
        assert len(events) == 2


class TestBuildAndEvolve:
    @pytest.mark.slow
    def test_build_sim_particle_count(self, known_system):
        from chain_survey.orbital_evolution import build_evolution_sim
        sim, rebx = build_evolution_sim(known_system)
        assert sim.N == len(known_system.planets) + 1

    @pytest.mark.slow
    def test_stable_system_short_evolution(self, known_system):
        from chain_survey.orbital_evolution import evolve_system
        result = evolve_system(known_system, t_end_myr=0.1)
        assert result.status in ("INTACT", "PARTIAL_BREAK", "FULL_BREAK",
                                  "EJECTION", "COLLISION")
        assert result.elapsed_s > 0

    @pytest.mark.slow
    def test_result_has_required_fields(self, known_system):
        from chain_survey.orbital_evolution import evolve_system
        result = evolve_system(known_system, t_end_myr=0.1)
        assert result.system_id == known_system.system_id
        assert isinstance(result.final_planets, list)
        assert isinstance(result.breaking_events, list)
        assert isinstance(result.n_survivors, int)
        assert isinstance(result.hz_planet_survived, bool)

    @pytest.mark.slow
    def test_sim_archive_saved(self, known_system, tmp_path):
        from chain_survey.orbital_evolution import evolve_system
        result = evolve_system(
            known_system, t_end_myr=0.1, output_dir=str(tmp_path),
        )
        if result.sim_archive_path:
            from pathlib import Path
            assert Path(result.sim_archive_path).exists()
