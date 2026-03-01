"""Subprocess isolation for chain survey pipeline stages.

Mirrors shared/sweep_runner.py pattern: single CLI entry point that
deserializes input JSON, runs the appropriate stage, and outputs
result JSON to stdout.

Usage: python -m chain_survey.survey_runner <stage> '<json>'

Stages:
  evolution  — Pipeline 1 (500 Myr orbital evolution)
  probe      — Perturbation probe (5K yr N-body)
  spin       — Pipeline 2 (5K yr spin dynamics)
"""

import json
import os
import subprocess
import sys
import time


def run_stage_safe(stage: str, input_json: str, timeout_s: int = 3600) -> dict:
    """Run a pipeline stage in a subprocess with timeout.

    Args:
        stage: One of "evolution", "probe", "spin".
        input_json: JSON string with stage-specific input.
        timeout_s: Timeout in seconds.

    Returns:
        Dict with result data, or error/timeout status.
    """
    t0 = time.time()

    try:
        # Ensure subprocess has LD_LIBRARY_PATH for REBOUND/REBOUNDx .so files
        env = os.environ.copy()
        venv_lib = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", ".venv", "lib", "python3.12", "site-packages"
        )
        if os.path.exists(venv_lib):
            env["LD_LIBRARY_PATH"] = f"{venv_lib}:{env.get('LD_LIBRARY_PATH', '')}"

        result = subprocess.run(
            [sys.executable, "-m", "chain_survey.survey_runner", stage, input_json],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )

        if result.returncode != 0:
            return {
                "status": "ERROR",
                "error_msg": result.stderr[:500] if result.stderr else f"Exit code {result.returncode}",
                "elapsed_s": time.time() - t0,
            }

        return json.loads(result.stdout)

    except subprocess.TimeoutExpired:
        return {
            "status": "TIMEOUT",
            "error_msg": f"Exceeded {timeout_s}s timeout",
            "elapsed_s": timeout_s,
        }

    except Exception as e:
        return {
            "status": "ERROR",
            "error_msg": str(e),
            "elapsed_s": time.time() - t0,
        }


def _run_evolution(input_data: dict) -> dict:
    """Run orbital evolution stage."""
    from .chain_types import SystemArchitecture
    from .orbital_evolution import evolve_system

    system = SystemArchitecture.from_dict(input_data["system"])
    t_end_myr = input_data.get("t_end_myr", 500.0)
    output_dir = input_data.get("output_dir")

    result = evolve_system(system, t_end_myr=t_end_myr, output_dir=output_dir)
    return result.to_dict()


def _run_probe(input_data: dict) -> dict:
    """Run perturbation probe stage."""
    from .perturbation_probe import run_probe

    result = run_probe(
        sim_archive_path=input_data["sim_archive_path"],
        hz_planet_idx=input_data["hz_planet_idx"],
        system_id=input_data["system_id"],
        n_years=input_data.get("n_years", 5000),
        thresholds=input_data.get("thresholds"),
    )
    return result.to_dict()


def _run_spin(input_data: dict) -> dict:
    """Run spin survey stage."""
    from .spin_survey import run_spin_survey

    result = run_spin_survey(
        sim_archive_path=input_data["sim_archive_path"],
        hz_planet_idx=input_data["hz_planet_idx"],
        system_id=input_data["system_id"],
        planet_mass_mearth=input_data.get("planet_mass_mearth", 1.0),
        planet_radius_rearth=input_data.get("planet_radius_rearth", 1.0),
        stellar_mass_msun=input_data.get("stellar_mass_msun", 0.15),
        tidal_q=input_data.get("tidal_q", 22),
        triaxiality=input_data.get("triaxiality", 3e-5),
        n_years=input_data.get("n_years", 5000),
    )
    return result.to_dict()


STAGES = {
    "evolution": _run_evolution,
    "probe": _run_probe,
    "spin": _run_spin,
}


def _cli_main():
    """CLI entry point for subprocess execution."""
    if len(sys.argv) != 3:
        print(
            "Usage: python -m chain_survey.survey_runner <stage> '<json>'",
            file=sys.stderr,
        )
        sys.exit(1)

    stage = sys.argv[1]
    if stage not in STAGES:
        print(f"Unknown stage: {stage}. Must be one of {list(STAGES.keys())}", file=sys.stderr)
        sys.exit(1)

    try:
        input_data = json.loads(sys.argv[2])
        result = STAGES[stage](input_data)

        def json_safe(obj):
            if hasattr(obj, "item"):
                return obj.item()
            if isinstance(obj, bool):
                return bool(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        print(json.dumps(result, default=json_safe))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _cli_main()
