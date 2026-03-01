"""Tests for spin_survey.py: flip-flop classification."""

import pytest

from chain_survey.spin_survey import classify_flipflop


class TestFlipFlopClassification:
    def test_flipflop_both_tl(self):
        fractions = {"TL_ZERO": 0.10, "TL_PI": 0.50, "SPINNING": 0.0, "PTB": 0.20}
        assert classify_flipflop(fractions) is True

    def test_no_flipflop_permanent_lock(self):
        fractions = {"TL_ZERO": 0.95, "TL_PI": 0.0, "SPINNING": 0.0, "PTB": 0.05}
        assert classify_flipflop(fractions) is False

    def test_no_flipflop_too_chaotic(self):
        fractions = {"TL_ZERO": 0.10, "TL_PI": 0.10, "SPINNING": 0.30, "PTB": 0.50}
        assert classify_flipflop(fractions) is False

    def test_no_flipflop_only_one_tl(self):
        fractions = {"TL_ZERO": 0.0, "TL_PI": 0.80, "SPINNING": 0.0, "PTB": 0.20}
        assert classify_flipflop(fractions) is False

    def test_no_flipflop_ptb_too_low(self):
        fractions = {"TL_ZERO": 0.10, "TL_PI": 0.87, "SPINNING": 0.0, "PTB": 0.03}
        assert classify_flipflop(fractions) is False

    def test_boundary_ptb_5_percent(self):
        fractions = {"TL_ZERO": 0.10, "TL_PI": 0.75, "SPINNING": 0.0, "PTB": 0.05}
        assert classify_flipflop(fractions) is True

    def test_boundary_ptb_30_percent(self):
        fractions = {"TL_ZERO": 0.10, "TL_PI": 0.50, "SPINNING": 0.0, "PTB": 0.30}
        assert classify_flipflop(fractions) is True

    def test_missing_keys_default_zero(self):
        fractions = {"PTB": 0.15}
        assert classify_flipflop(fractions) is False
