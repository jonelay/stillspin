"""Grid evaluation and pymoo Problem class for obliquity parameter sweep."""

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from physics import integrate_obliquity, measure_oscillation


class MoonSweepProblem(ElementwiseProblem):
    """2 variables (moon mass, moon distance) → 2 objectives (period error, amplitude error).

    Variables:
        x[0] = log10(moon_mass / M_Earth)
        x[1] = moon_distance / R_planet

    Objectives (minimize):
        F[0] = |period - target_period|  (years)
        F[1] = |amplitude - target_amplitude|  (degrees)
    """

    def __init__(self, config):
        sw = config['sweep']
        self.mass_min = sw['moon_mass']['min']
        self.mass_max = sw['moon_mass']['max']
        self.dist_min = sw['moon_distance']['min']
        self.dist_max = sw['moon_distance']['max']

        # Bounds in log-mass and linear-distance
        xl = np.array([np.log10(self.mass_min), self.dist_min])
        xu = np.array([np.log10(self.mass_max), self.dist_max])

        super().__init__(n_var=2, n_obj=2, xl=xl, xu=xu)

        self.system = config['system']
        self.integration = config['integration']

        targets = config['targets']
        self.target_period = (targets['period_yr']['min'] + targets['period_yr']['max']) / 2.0
        self.target_amplitude = (targets['amplitude_deg']['min'] + targets['amplitude_deg']['max']) / 2.0

    def _evaluate(self, x, out, *args, **kwargs):
        log_mass, dist = x
        mass = 10.0 ** log_mass

        params = dict(self.system)
        params['moon_mass_mearth'] = mass
        params['moon_distance_rp'] = dist

        result = integrate_obliquity(
            params,
            t_stop_yr=self.integration['t_stop_yr'],
            method=self.integration.get('method', 'RK45'),
            rtol=self.integration.get('rtol', 1e-9),
            atol=self.integration.get('atol', 1e-12),
        )

        if not result.get('success', False) or len(result['t_yr']) == 0:
            out["F"] = [1e6, 1e6]
            return

        period, amplitude = measure_oscillation(result['t_yr'], result['obliquity_deg'])

        period_err = abs(period - self.target_period) if np.isfinite(period) else 1e6
        amp_err = abs(amplitude - self.target_amplitude) if np.isfinite(amplitude) else 1e6

        out["F"] = [period_err, amp_err]


def _build_axis(cfg):
    """Build a 1D array from an axis config dict with min/max/points/scale."""
    if cfg.get('scale', 'linear') == 'log':
        return np.logspace(np.log10(cfg['min']), np.log10(cfg['max']), cfg['points'])
    else:
        return np.linspace(cfg['min'], cfg['max'], cfg['points'])


def build_grid(config):
    """Build grid arrays for all sweep axes.

    Returns (masses, dists) for backward compat if only 2D axes present,
    or (incl_vals, kerb_vals, masses, dists) for 4D.
    Always returns the 4-tuple form now.
    """
    sw = config['sweep']

    # New axes — default to single-point if absent (backward compat)
    if 'mutual_inclination_deg' in sw:
        incl_vals = _build_axis(sw['mutual_inclination_deg'])
    else:
        incl_vals = np.array([config['system'].get('mutual_inclination_deg', 5.0)])

    if 'kerberos_distance_au' in sw:
        kerb_vals = _build_axis(sw['kerberos_distance_au'])
    else:
        kerb_vals = np.array([config['system'].get('kerberos_distance_au', 8.0)])

    masses = _build_axis(sw['moon_mass'])
    dists = _build_axis(sw['moon_distance'])

    return incl_vals, kerb_vals, masses, dists


def evaluate_grid(config):
    """Run the full 4D grid sweep.

    Returns dict with:
        incl_vals, kerb_vals, masses, distances: 1D arrays for each axis
        period, amplitude, alpha_over_s, alpha, s: 4D arrays
            [n_incl × n_kerb × n_mass × n_dist]
        i_forced_deg: 4D array of forced inclination (for model-validity annotation)
    """
    incl_vals, kerb_vals, masses, dists = build_grid(config)
    system = config['system']
    integ = config['integration']

    ni, nk, nm, nd = len(incl_vals), len(kerb_vals), len(masses), len(dists)
    shape = (ni, nk, nm, nd)

    period = np.full(shape, np.nan)
    amplitude = np.full(shape, np.nan)
    alpha_arr = np.full(shape, np.nan)
    s_arr = np.full(shape, np.nan)
    alpha_over_s = np.full(shape, np.nan)
    i_forced_deg = np.full(shape, np.nan)

    total = ni * nk * nm * nd
    cell = 0
    for ii, incl in enumerate(incl_vals):
        for ik, kerb in enumerate(kerb_vals):
            for im, m in enumerate(masses):
                for id_, d in enumerate(dists):
                    cell += 1
                    print(f"  [{cell}/{total}] I_mut={incl:.0f}° a_kerb={kerb:.1f}AU "
                          f"mass={m:.4f}M⊕ dist={d:.1f}Rp", end="")

                    params = dict(system)
                    params['mutual_inclination_deg'] = incl
                    params['kerberos_distance_au'] = kerb
                    params['moon_mass_mearth'] = m
                    params['moon_distance_rp'] = d

                    result = integrate_obliquity(
                        params,
                        t_stop_yr=integ['t_stop_yr'],
                        method=integ.get('method', 'RK45'),
                        rtol=integ.get('rtol', 1e-9),
                        atol=integ.get('atol', 1e-12),
                    )

                    alpha_arr[ii, ik, im, id_] = result['alpha']
                    s_arr[ii, ik, im, id_] = result['s']
                    alpha_over_s[ii, ik, im, id_] = result['alpha_over_s']
                    i_forced_deg[ii, ik, im, id_] = result.get('i_forced_deg', np.nan)

                    if result.get('success', False) and len(result['t_yr']) > 0:
                        p, a = measure_oscillation(result['t_yr'], result['obliquity_deg'])
                        period[ii, ik, im, id_] = p
                        amplitude[ii, ik, im, id_] = a
                        print(f" → P={p:.1f}yr A={a:.2f}°")
                    else:
                        print(f" → FAILED: {result.get('message', 'unknown')}")

    return {
        'incl_vals': incl_vals,
        'kerb_vals': kerb_vals,
        'masses': masses,
        'distances': dists,
        'period': period,
        'amplitude': amplitude,
        'alpha': alpha_arr,
        's': s_arr,
        'alpha_over_s': alpha_over_s,
        'i_forced_deg': i_forced_deg,
    }
