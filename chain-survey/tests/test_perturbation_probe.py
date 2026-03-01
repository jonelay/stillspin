"""Tests for perturbation_probe.py: metrics, spectrum, filter."""

import numpy as np
import pytest

from chain_survey.perturbation_probe import (
    apply_filter,
    compute_spectrum,
    extract_perturbation_metrics,
)

from shared.constants import YEAR


class TestPerturbationMetrics:
    def test_rms_perturbation_positive(self):
        t = np.linspace(0, 1000 * YEAR, 10000)
        n = 3.8e-6 * (1 + 0.01 * np.sin(2 * np.pi * t / (15 * YEAR)))
        e = 0.05 + 0.005 * np.sin(2 * np.pi * t / (15 * YEAR))
        metrics = extract_perturbation_metrics(t, e, n)
        assert metrics["rms_dn_over_n"] > 0

    def test_rms_perturbation_reasonable(self):
        t = np.linspace(0, 1000 * YEAR, 10000)
        n = 3.8e-6 * (1 + 0.01 * np.sin(2 * np.pi * t / (15 * YEAR)))
        e = 0.05 + 0.005 * np.sin(2 * np.pi * t / (15 * YEAR))
        metrics = extract_perturbation_metrics(t, e, n)
        assert metrics["rms_dn_over_n"] < 1.0

    def test_max_e_correct(self):
        t = np.linspace(0, 100 * YEAR, 1000)
        e = np.array([0.01, 0.05, 0.1, 0.03] * 250)
        n = np.ones_like(t) * 3.8e-6
        metrics = extract_perturbation_metrics(t, e, n)
        assert metrics["max_e"] == pytest.approx(0.1)

    def test_mean_n_correct(self):
        t = np.linspace(0, 100 * YEAR, 1000)
        n = np.ones(1000) * 3.8e-6
        e = np.ones(1000) * 0.05
        metrics = extract_perturbation_metrics(t, e, n)
        assert metrics["mean_n_rad_s"] == pytest.approx(3.8e-6)


class TestSpectrum:
    def test_returns_top_n_peaks(self):
        t = np.linspace(0, 5000 * YEAR, 50000)
        n = 3.8e-6 * (1 + 0.01 * np.sin(2 * np.pi * t / (15 * YEAR)))
        periods = compute_spectrum(t, n, n_peaks=5)
        assert len(periods) <= 5

    def test_periods_positive(self):
        t = np.linspace(0, 5000 * YEAR, 50000)
        n = 3.8e-6 * (1 + 0.01 * np.sin(2 * np.pi * t / (15 * YEAR)))
        periods = compute_spectrum(t, n)
        assert all(p > 0 for p in periods)

    def test_detects_known_period(self):
        t = np.linspace(0, 5000 * YEAR, 50000)
        period_yr = 15.0
        n = 3.8e-6 * (1 + 0.05 * np.sin(2 * np.pi * t / (period_yr * YEAR)))
        periods = compute_spectrum(t, n, n_peaks=3)
        if periods:
            # Dominant period should be close to 15 years
            assert min(abs(p - period_yr) for p in periods) < 2.0


class TestFilter:
    def test_none_thresholds_always_pass(self):
        assert apply_filter(0.001, thresholds=None) == "PASS"
        assert apply_filter(0.1, thresholds=None) == "PASS"

    def test_pass_for_moderate(self):
        thresholds = {"min": 0.001, "max": 0.1}
        assert apply_filter(0.01, thresholds) == "PASS"

    def test_reject_weak(self):
        thresholds = {"min": 0.001, "max": 0.1}
        assert apply_filter(0.0001, thresholds) == "REJECT_WEAK"

    def test_reject_strong(self):
        thresholds = {"min": 0.001, "max": 0.1}
        assert apply_filter(0.5, thresholds) == "REJECT_STRONG"
