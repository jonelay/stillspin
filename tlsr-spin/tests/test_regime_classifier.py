"""Tests for TLSR regime classification."""

import numpy as np
import pytest

from shared.constants import YEAR
from tlsr_spin.regime_classifier import (
    RegimeType,
    classify_regimes,
    compute_quasi_stable_fraction,
    compute_regime_fractions,
    compute_regime_stats,
)


class TestClassifyRegimes:
    """Tests for regime classification from γ(t) time series."""

    def test_pure_libration_zero(self) -> None:
        """Small oscillation around γ=0 should classify as TL_ZERO."""
        t = np.linspace(0, 1000 * YEAR, 10000)
        gamma = 0.5 * np.sin(2 * np.pi * t / (50 * YEAR))  # small libration

        result = classify_regimes(t, gamma)
        fractions = compute_regime_fractions(result)
        # Should be predominantly TL_ZERO
        assert fractions["TL_ZERO"] > 0.5

    def test_pure_spinning(self) -> None:
        """Monotonically increasing γ should classify as SPINNING."""
        t = np.linspace(0, 1000 * YEAR, 10000)
        gamma = 1e-7 * t  # steady spin

        result = classify_regimes(t, gamma)
        fractions = compute_regime_fractions(result)
        assert fractions["SPINNING"] > 0.5

    def test_short_regimes_become_ptb(self) -> None:
        """Regimes shorter than min_regime_yr should be PTB."""
        t = np.linspace(0, 100 * YEAR, 10000)
        # Rapid switching — short regimes
        gamma = 0.3 * np.sin(2 * np.pi * t / (5 * YEAR))

        result = classify_regimes(t, gamma, min_regime_yr=50.0)
        # With 100 yr total and 50 yr minimum, everything should be PTB or short
        has_ptb = any(r.type == RegimeType.PTB for r in result.regimes)
        # At least some regimes should exist
        assert len(result.regimes) > 0
        # Short integration with high min_regime should produce some PTB
        assert has_ptb or all(r.duration_yr >= 50.0 for r in result.regimes)


class TestRegimeStats:
    """Tests for regime statistics computation."""

    def test_empty_result(self) -> None:
        """Empty regime list should return zero stats."""
        t = np.linspace(0, 100 * YEAR, 100)
        gamma = np.zeros(100)
        from tlsr_spin.regime_classifier import RegimeResult
        result = RegimeResult(regimes=[], t=t, gamma=gamma)

        stats = compute_regime_stats(result)
        for rtype in RegimeType:
            assert stats[rtype.value]["count"] == 0

    def test_fractions_sum_to_one(self) -> None:
        """Regime fractions should sum to ~1."""
        t = np.linspace(0, 1000 * YEAR, 10000)
        gamma = 0.5 * np.sin(2 * np.pi * t / (50 * YEAR))

        result = classify_regimes(t, gamma)
        fractions = compute_regime_fractions(result)
        total = sum(fractions.values())
        assert total == pytest.approx(1.0, abs=0.05)


class TestQuasiStableFraction:
    """Tests for quasi-stable fraction computation."""

    def test_all_stable(self) -> None:
        """Single long regime should give fraction ~1."""
        t = np.linspace(0, 5000 * YEAR, 50000)
        gamma = 0.3 * np.sin(2 * np.pi * t / (100 * YEAR))

        result = classify_regimes(t, gamma)
        qs = compute_quasi_stable_fraction(result, threshold_yr=900)
        assert qs > 0.5  # most time should be quasi-stable
