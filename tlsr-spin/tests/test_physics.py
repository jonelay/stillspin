"""Tests for TLSR spin-orbit physics functions."""

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from tlsr_spin.physics import (
    eccentricity_function,
    omega_s_squared,
    spin_ode,
    tidal_epsilon,
)


class TestEccentricityFunction:
    """Tests for H(e) = 1 - 5e²/2 + 13e⁴/16."""

    def test_circular_orbit(self) -> None:
        """H(0) = 1 for a circular orbit."""
        assert eccentricity_function(0.0) == pytest.approx(1.0)

    def test_small_eccentricity(self) -> None:
        """H(0.1) ≈ 1 - 0.025 + tiny = 0.975..."""
        h = eccentricity_function(0.1)
        expected = 1.0 - 2.5 * 0.01 + (13.0 / 16.0) * 0.0001
        assert h == pytest.approx(expected)

    def test_moderate_eccentricity(self) -> None:
        """H(0.3) hand calculation."""
        e = 0.3
        expected = 1.0 - 2.5 * e**2 + (13.0 / 16.0) * e**4
        assert eccentricity_function(e) == pytest.approx(expected)

    def test_decreases_with_eccentricity(self) -> None:
        """H(e) should decrease for moderate eccentricities."""
        assert eccentricity_function(0.2) < eccentricity_function(0.0)

    def test_symmetric(self) -> None:
        """H(e) depends on e², so H(-e) = H(e) (though e >= 0 physically)."""
        assert eccentricity_function(0.3) == pytest.approx(eccentricity_function(-0.3))


class TestOmegaSSquared:
    """Tests for ω_s² = 3 n² (B-A)/C |H(e)|."""

    def test_circular_orbit(self) -> None:
        """For e=0: ω_s² = 3 n² (B-A)/C."""
        n = 1e-6  # rad/s
        triax = 1e-5
        result = omega_s_squared(n, triax, 0.0)
        expected = 3.0 * n**2 * triax
        assert result == pytest.approx(expected)

    def test_scales_with_n_squared(self) -> None:
        """ω_s² ∝ n²."""
        triax = 1e-5
        w1 = omega_s_squared(1e-6, triax, 0.0)
        w2 = omega_s_squared(2e-6, triax, 0.0)
        assert w2 == pytest.approx(4.0 * w1)

    def test_scales_with_triaxiality(self) -> None:
        """ω_s² ∝ (B-A)/C."""
        n = 1e-6
        w1 = omega_s_squared(n, 1e-5, 0.0)
        w2 = omega_s_squared(n, 1e-4, 0.0)
        assert w2 == pytest.approx(10.0 * w1)


class TestTidalEpsilon:
    """Tests for tidal dissipation coefficient."""

    def test_positive(self) -> None:
        """ε should be positive for physical parameters."""
        eps = tidal_epsilon(
            m_star=1.989e30 * 0.36,
            m_planet=5.972e24,
            r_planet=6.371e6,
            a=0.18 * 1.496e11,
            tidal_q=10,
            omega_spin=1e-6,
        )
        assert eps > 0

    def test_inversely_proportional_to_q(self) -> None:
        """ε ∝ 1/Q."""
        kwargs = dict(
            m_star=1.989e30 * 0.36,
            m_planet=5.972e24,
            r_planet=6.371e6,
            a=0.18 * 1.496e11,
            omega_spin=1e-6,
        )
        eps_q10 = tidal_epsilon(**kwargs, tidal_q=10)
        eps_q100 = tidal_epsilon(**kwargs, tidal_q=100)
        assert eps_q10 == pytest.approx(10.0 * eps_q100, rel=1e-10)


class TestSpinODE:
    """Tests for the spin-orbit ODE right-hand side."""

    @pytest.fixture()
    def constant_splines(self) -> tuple[CubicSpline, CubicSpline]:
        """Create constant e(t) and n(t) splines for testing."""
        t = np.linspace(0, 1e10, 100)
        e = np.full_like(t, 0.05)
        n = np.full_like(t, 1.57e-6)
        return CubicSpline(t, e), CubicSpline(t, n)

    def test_equilibrium_at_zero(self, constant_splines: tuple) -> None:
        """At γ=0, γ̇=0, sin(2γ)=0, so γ̈ ≈ -ṅ (≈0 for constant n)."""
        e_spl, n_spl = constant_splines
        result = spin_ode(5e9, np.array([0.0, 0.0]), e_spl, n_spl, 1e-5, 0.0)
        assert abs(result[0]) < 1e-30  # γ̇ = 0
        assert abs(result[1]) < 1e-20  # γ̈ ≈ 0 (ṅ ≈ 0 for constant n)

    def test_restoring_force(self, constant_splines: tuple) -> None:
        """Displaced from γ=0, should get a restoring γ̈ toward 0."""
        e_spl, n_spl = constant_splines
        # Small positive displacement
        result = spin_ode(5e9, np.array([0.1, 0.0]), e_spl, n_spl, 1e-5, 0.0)
        # γ̈ should be negative (restoring toward 0)
        assert result[1] < 0

    def test_damping(self, constant_splines: tuple) -> None:
        """With ε > 0 and γ̇ > 0, damping term -εγ̇ < 0."""
        e_spl, n_spl = constant_splines
        eps = 1e-10
        gamma_dot = 1e-8
        result = spin_ode(
            5e9, np.array([0.0, gamma_dot]), e_spl, n_spl, 1e-5, eps,
        )
        # γ̈ should include -ε * γ̇ contribution (negative)
        result_no_damp = spin_ode(
            5e9, np.array([0.0, gamma_dot]), e_spl, n_spl, 1e-5, 0.0,
        )
        assert result[1] < result_no_damp[1]

    def test_returns_two_values(self, constant_splines: tuple) -> None:
        """ODE RHS returns [γ̇, γ̈]."""
        e_spl, n_spl = constant_splines
        result = spin_ode(5e9, np.array([0.5, 1e-8]), e_spl, n_spl, 1e-5, 0.0)
        assert len(result) == 2

    def test_first_element_is_gamma_dot(self, constant_splines: tuple) -> None:
        """First element of RHS should be γ̇ (passed through)."""
        e_spl, n_spl = constant_splines
        gamma_dot = 3.14e-8
        result = spin_ode(
            5e9, np.array([0.0, gamma_dot]), e_spl, n_spl, 1e-5, 0.0,
        )
        assert result[0] == pytest.approx(gamma_dot)
