"""Tests for pipeline data flow: stage linking, resume, serialization."""

import json
import pytest

from chain_survey.chain_types import (
    OrbitalResult,
    PerturbationProbe,
    SpinSurveyResult,
    SystemArchitecture,
)


class TestSystemIdLinking:
    def test_system_id_links_all_stages(
        self, known_system, mock_orbital_result, mock_probe_pass,
    ):
        sid = known_system.system_id
        assert mock_orbital_result.system_id == sid
        assert mock_probe_pass.system_id == sid

        spin_result = SpinSurveyResult(
            system_id=sid, tidal_q=22, triaxiality=3e-5,
            fractions={"TL_ZERO": 0.1, "TL_PI": 0.5, "SPINNING": 0.0, "PTB": 0.2},
            is_flipflop=True, episodes=[], status="OK", elapsed_s=5.0,
        )
        assert spin_result.system_id == sid

    def test_filtered_systems_subset_of_probed(self, mock_probe_pass, mock_probe_reject):
        all_probes = [mock_probe_pass, mock_probe_reject]
        passed = [p for p in all_probes if p.filter_verdict == "PASS"]
        assert len(passed) == 1
        assert passed[0].system_id == mock_probe_pass.system_id


class TestJSONLRoundtrip:
    def test_system_jsonl_roundtrip(self, known_system):
        line = json.dumps(known_system.to_dict())
        restored = SystemArchitecture.from_dict(json.loads(line))
        assert restored.system_id == known_system.system_id

    def test_orbital_result_jsonl_roundtrip(self, mock_orbital_result):
        line = json.dumps(mock_orbital_result.to_dict())
        restored = OrbitalResult.from_dict(json.loads(line))
        assert restored.system_id == mock_orbital_result.system_id
        assert restored.status == mock_orbital_result.status

    def test_probe_jsonl_roundtrip(self, mock_probe_pass):
        line = json.dumps(mock_probe_pass.to_dict())
        restored = PerturbationProbe.from_dict(json.loads(line))
        assert restored.filter_verdict == "PASS"

    def test_spin_result_jsonl_roundtrip(self):
        r = SpinSurveyResult(
            system_id="test123", tidal_q=22, triaxiality=3e-5,
            fractions={"TL_ZERO": 0.1, "TL_PI": 0.5, "SPINNING": 0.1, "PTB": 0.2},
            is_flipflop=True, episodes=[{"type": "PTB", "duration_yr": 1.2}],
            status="OK", elapsed_s=5.0,
        )
        line = json.dumps(r.to_dict())
        restored = SpinSurveyResult.from_dict(json.loads(line))
        assert restored.is_flipflop is True


class TestCalibrationMode:
    def test_calibration_batch_no_filter(self):
        from chain_survey.perturbation_probe import apply_filter

        # In calibration mode (no thresholds), everything passes
        for rms in [0.0001, 0.001, 0.01, 0.1, 1.0]:
            assert apply_filter(rms, thresholds=None) == "PASS"
