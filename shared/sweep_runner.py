"""Timeout-safe execution wrapper for sweep configurations.

Addresses v3.2 infrastructure failure: 23-hour hang due to pathological
configs with no timeout mechanism.

Design v2 (2026-02-13):
- subprocess.run with timeout: Actually kills hung processes (unlike ProcessPoolExecutor)
- JSON stdin/stdout: Config passed as JSON, result returned as JSON
- CLI entry point: `python -m shared.sweep_runner '{"q": 20, ...}'`

Previous design (v1) used ProcessPoolExecutor.future.result(timeout=X) which
only times out the wait, not the actual worker process.
"""

import json
import subprocess
import sys
import time

from shared.sweep_types import SweepConfig, SweepResult


def _run_single_impl(config: SweepConfig) -> dict:
    """Internal implementation that runs in subprocess.

    This function must be at module level for pickling by ProcessPoolExecutor.
    """
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from shared.analysis import compute_ptb_quality_score
    from shared.constants import L_SUN, M_EARTH, M_SUN, R_EARTH
    from thermal_sweep.ebm import tidal_lock_temperatures
    from tlsr_spin.sweep import run_single

    cli_config = {
        "BIPOLARIS_DISTANCE_AU": config.distance_au,
        "SURFACE_ALBEDO": config.albedo,
        "ATMO_CO2_FRACTION_MID": config.co2_pct / 100,
        "STAR_MASS_MSUN": 0.15,
        "STAR_MASS": 0.15 * M_SUN,
        "STAR_LUMINOSITY_LSUN": 0.0022,
        "STAR_LUMINOSITY": 0.0022 * L_SUN,
        "STAR_TEFF": 2900,
        "BIPOLARIS_MASS": 1.0 * M_EARTH,
        "BIPOLARIS_RADIUS": 1.06 * R_EARTH,
    }

    temps = tidal_lock_temperatures(
        distance_au=config.distance_au,
        luminosity_lsun=cli_config["STAR_LUMINOSITY_LSUN"],
        albedo=config.albedo,
        co2_fraction=config.co2_pct / 100,
    )

    result = run_single(
        scenario_name="A",
        triaxiality=config.triax,
        tidal_q=config.q,
        n_orbits=config.n_orbits,
        cli_overrides=cli_config,
        quiet=True,
    )

    fractions = result["fractions"]
    quality = compute_ptb_quality_score(
        ptb_frac=fractions.get("PTB", 0),
        tl_zero_frac=fractions.get("TL_ZERO", 0),
        tl_pi_frac=fractions.get("TL_PI", 0),
        t_term=temps["t_terminator"],
    )

    # Extract episode data from regime_result
    episodes = None
    regime_result = result.get("regime_result")
    if regime_result is not None:
        from shared.constants import YEAR
        from shared.sweep_types import Episode

        episodes = []
        regimes = regime_result.regimes

        for i, regime in enumerate(regimes):
            neighbors = None
            if regime.type.value == "PTB":
                neighbors = {
                    "before": regimes[i - 1].type.value if i > 0 else None,
                    "after": regimes[i + 1].type.value if i < len(regimes) - 1 else None,
                }

            episodes.append(
                Episode(
                    type=regime.type.value,
                    duration_yr=regime.duration_yr,
                    t_start_yr=regime.t_start / YEAR,
                    t_end_yr=regime.t_end / YEAR,
                    neighbors=neighbors,
                )
            )

    return {
        "fractions": fractions,
        "temps": {
            "t_sub": round(temps["t_substellar"], 1),
            "t_term": round(temps["t_terminator"], 1),
            "t_anti": round(temps["t_antistellar"], 1),
        },
        "quality": quality,
        "episodes": episodes,
    }


def run_config_safe(config: SweepConfig, timeout_s: int = 600) -> SweepResult:
    """Run single config with true timeout via subprocess.

    Uses subprocess.run with timeout to guarantee the process is killed
    if it exceeds the timeout. This fixes the v3.2 hang issue where
    ProcessPoolExecutor.future.result(timeout) only times out the wait.

    Args:
        config: Validated sweep configuration.
        timeout_s: Timeout in seconds (default 600 = 10 min).

    Returns:
        SweepResult with status OK, TIMEOUT, or ERROR.
    """
    t0 = time.time()

    try:
        result = subprocess.run(
            [sys.executable, "-m", "shared.sweep_runner", json.dumps(config.to_dict())],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=None,  # Use current working directory
        )

        if result.returncode != 0:
            return SweepResult(
                config=config,
                status="ERROR",
                fractions=None,
                temps=None,
                quality=None,
                episodes=None,
                elapsed_s=time.time() - t0,
                error_msg=result.stderr[:500] if result.stderr else f"Exit code {result.returncode}",
            )

        raw = json.loads(result.stdout)

        # Reconstruct episodes from dict
        episodes = None
        if raw.get("episodes"):
            from shared.sweep_types import Episode
            episodes = [Episode.from_dict(ep) for ep in raw["episodes"]]

        return SweepResult(
            config=config,
            status="OK",
            fractions=raw["fractions"],
            temps=raw["temps"],
            quality=raw["quality"],
            episodes=episodes,
            elapsed_s=time.time() - t0,
        )

    except subprocess.TimeoutExpired:
        return SweepResult(
            config=config,
            status="TIMEOUT",
            fractions=None,
            temps=None,
            quality=None,
            episodes=None,
            elapsed_s=timeout_s,
            error_msg=f"Exceeded {timeout_s}s timeout (process killed)",
        )

    except Exception as e:
        return SweepResult(
            config=config,
            status="ERROR",
            fractions=None,
            temps=None,
            quality=None,
            episodes=None,
            elapsed_s=time.time() - t0,
            error_msg=str(e),
        )


def _cli_main():
    """CLI entry point for subprocess execution.

    Usage: python -m shared.sweep_runner '{"q": 20, "distance_au": 0.07, ...}'

    Reads config JSON from argv[1], runs simulation, outputs result JSON to stdout.
    """
    if len(sys.argv) != 2:
        print("Usage: python -m shared.sweep_runner '<config_json>'", file=sys.stderr)
        sys.exit(1)

    try:
        config_dict = json.loads(sys.argv[1])
        config = SweepConfig.from_dict(config_dict)
        raw = _run_single_impl(config)

        # Serialize episodes to dict
        if raw.get("episodes"):
            raw["episodes"] = [ep.to_dict() for ep in raw["episodes"]]

        # Convert numpy/bool types for JSON
        def json_safe(obj):
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            if isinstance(obj, bool):
                return bool(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        print(json.dumps(raw, default=json_safe))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _cli_main()
