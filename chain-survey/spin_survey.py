"""Pipeline 2: Spin dynamics survey wrapping tlsr_spin.

Loads post-evolution REBOUND simulation, extracts orbital perturbations,
runs spin integration, and classifies flip-flop behavior.
"""

import time
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*reset function pointers.*", category=RuntimeWarning)

from shared.constants import AU, M_EARTH, M_SUN, R_EARTH, YEAR

from .chain_types import SpinSurveyResult


def classify_flipflop(fractions: dict) -> bool:
    """True if system shows flip-flop behavior.

    Criteria: PTB in [5%, 30%] AND TL_ZERO > 1% AND TL_PI > 1%.
    """
    ptb = fractions.get("PTB", 0)
    tl_zero = fractions.get("TL_ZERO", 0)
    tl_pi = fractions.get("TL_PI", 0)
    return 0.05 <= ptb <= 0.30 and tl_zero > 0.01 and tl_pi > 0.01


def run_spin_survey(
    sim_archive_path: str,
    hz_planet_idx: int,
    system_id: str,
    planet_mass_mearth: float = 1.0,
    planet_radius_rearth: float = 1.0,
    stellar_mass_msun: float = 0.15,
    tidal_q: int = 22,
    triaxiality: float = 3e-5,
    n_years: int = 5000,
) -> SpinSurveyResult:
    """Full spin dynamics pipeline for one HZ planet.

    Reuses existing tlsr_spin infrastructure:
    1. Load sim from archive
    2. integrate_and_extract → e(t), n(t)
    3. integrate_spin → gamma(t)
    4. classify_regimes → regime fractions
    """
    t0 = time.time()

    try:
        import rebound

        from tlsr_spin.nbody import integrate_and_extract
        from tlsr_spin.regime_classifier import classify_regimes, compute_regime_fractions
        from tlsr_spin.spin_integrator import integrate_spin

        sim = rebound.Simulation(sim_archive_path)

        # Compute n_orbits from n_years and planet period
        orb = sim.particles[hz_planet_idx].orbit()
        p_orb_yr = orb.P
        n_orbits = int(n_years / p_orb_yr) if p_orb_yr > 0 else 5000

        # Extract orbital perturbations
        nbody = integrate_and_extract(sim, hz_planet_idx, n_orbits)

        # Physical parameters
        m_star = stellar_mass_msun * M_SUN
        m_planet = planet_mass_mearth * M_EARTH
        r_planet = planet_radius_rearth * R_EARTH
        a_mean = float(np.mean(nbody["a"])) * AU

        # Spin integration
        spin_result = integrate_spin(
            times=nbody["t"],
            e_t=nbody["e"],
            n_t=nbody["n"],
            m_star=m_star,
            m_planet=m_planet,
            r_planet=r_planet,
            a_mean=a_mean,
            tidal_q=tidal_q,
            triaxiality=triaxiality,
            n_output=n_years,
        )

        # Regime classification
        regime_result = classify_regimes(spin_result["t"], spin_result["gamma"])
        fractions = compute_regime_fractions(regime_result)

        is_ff = classify_flipflop(fractions)

        # Extract episodes
        episodes = []
        for regime in regime_result.regimes:
            ep = {
                "type": regime.type.value,
                "duration_yr": regime.duration_yr,
                "t_start_yr": regime.t_start / YEAR,
                "t_end_yr": regime.t_end / YEAR,
            }
            episodes.append(ep)

        return SpinSurveyResult(
            system_id=system_id,
            tidal_q=tidal_q,
            triaxiality=triaxiality,
            fractions=fractions,
            is_flipflop=is_ff,
            episodes=episodes,
            status="OK",
            elapsed_s=time.time() - t0,
        )

    except Exception as e:
        return SpinSurveyResult(
            system_id=system_id,
            tidal_q=tidal_q,
            triaxiality=triaxiality,
            fractions=None,
            is_flipflop=False,
            episodes=None,
            status="ERROR",
            elapsed_s=time.time() - t0,
            error_msg=str(e),
        )
