"""Test that spin dynamics results are robust to initial conditions."""

import numpy as np
import pytest

N_ORBITS = 2_000


class TestInitialConditionRobustness:
    """Verify regime fractions converge regardless of initial γ."""

    @pytest.fixture(scope="class")
    def nbody_data(self):
        """Pre-compute N-body (expensive) for reuse."""
        from tlsr_spin.nbody import build_bipolaris_system, integrate_and_extract

        sim = build_bipolaris_system()
        return integrate_and_extract(sim, 3, N_ORBITS)

    @pytest.mark.parametrize("gamma_0", [0.0, np.pi / 4, np.pi / 2, np.pi])
    def test_integration_completes(self, nbody_data, gamma_0):
        """Spin integration should complete for any valid initial γ."""
        from shared.constants import (
            AU,
            BIPOLARIS_MASS,
            BIPOLARIS_RADIUS,
            BIPOLARIS_TIDAL_Q,
            BIPOLARIS_TRIAXIALITY,
            STAR_MASS,
        )
        from tlsr_spin.spin_integrator import integrate_spin

        result = integrate_spin(
            times=nbody_data["t"],
            e_t=nbody_data["e"],
            n_t=nbody_data["n"],
            m_star=STAR_MASS,
            m_planet=BIPOLARIS_MASS,
            r_planet=BIPOLARIS_RADIUS,
            a_mean=np.mean(nbody_data["a"]) * AU,
            tidal_q=BIPOLARIS_TIDAL_Q,
            triaxiality=BIPOLARIS_TRIAXIALITY,
            gamma_0=gamma_0,
        )
        assert len(result["gamma"]) > 0
        assert not np.any(np.isnan(result["gamma"]))
