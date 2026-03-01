#!/usr/bin/env python3
"""
Parameter sweep for TLSR spin dynamics.

Sweeps across system configurations (from shared/scenarios.py) and
TLSR-specific parameters (triaxiality, tidal Q). Generates summary
report and comparison figures.

Replaces DayNite's HPC queue scripts with a single CLI tool.
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from shared.constants import (
    AU,
    BIPOLARIS_DISTANCE_AU,
    BIPOLARIS_MASS,
    BIPOLARIS_RADIUS,
    M_SUN,
    STAR_MASS,
    YEAR,
)
from shared.paths import output_dir_for, results_dir
from shared.scenarios import SCENARIOS, get_scenario

from tlsr_spin.nbody import build_bipolaris_system, integrate_and_extract
from tlsr_spin.plots import plot_regime_pie, plot_regime_summary, plot_spin_history
from tlsr_spin.regime_classifier import (
    classify_regimes,
    compute_quasi_stable_fraction,
    compute_regime_fractions,
    compute_regime_stats,
)
from tlsr_spin.spin_integrator import integrate_spin


# Default sweep parameters
DEFAULT_TRIAXIALITIES = [1e-6, 1e-5, 1e-4]
DEFAULT_TIDAL_QS = [10, 50, 100, 500, 1000]  # expanded for TLSR exploration


def run_single(
    scenario_name: str,
    triaxiality: float,
    tidal_q: int,
    n_orbits: int,
    planet_idx: int = 3,
    cli_overrides: dict | None = None,
    quiet: bool = False,
) -> dict:
    """Run a single TLSR spin simulation.

    Args:
        scenario_name: Scenario key from shared/scenarios.py.
        triaxiality: (B-A)/C parameter.
        tidal_q: Tidal quality factor.
        n_orbits: Number of N-body orbits to integrate.
        planet_idx: Target planet (3 = Bipolaris).
        cli_overrides: Additional overrides from CLI (e.g., --a-override).
        quiet: Suppress progress prints (for parallel workers).

    Returns:
        Dict with results: fractions, stats, quasi_stable_fraction, etc.
    """
    overrides = get_scenario(scenario_name)
    if cli_overrides:
        overrides.update(cli_overrides)

    # N-body
    if not quiet:
        print(f"  N-body: {n_orbits:,} orbits...")
    sim = build_bipolaris_system(overrides)
    nbody = integrate_and_extract(sim, planet_idx, n_orbits)

    # Planet parameters (with overrides)
    m_star = overrides.get("STAR_MASS", STAR_MASS)
    m_planet = overrides.get("BIPOLARIS_MASS", BIPOLARIS_MASS)
    r_planet = overrides.get("BIPOLARIS_RADIUS", BIPOLARIS_RADIUS)
    a_mean = float(np.mean(nbody["a"])) * AU

    # Spin integration
    if not quiet:
        print(f"  Spin ODE: triax={triaxiality:.0e}, Q={tidal_q}...")
    spin = integrate_spin(
        times=nbody["t"],
        e_t=nbody["e"],
        n_t=nbody["n"],
        m_star=m_star,
        m_planet=m_planet,
        r_planet=r_planet,
        a_mean=a_mean,
        tidal_q=tidal_q,
        triaxiality=triaxiality,
    )

    # Regime classification
    result = classify_regimes(spin["t"], spin["gamma"])
    stats = compute_regime_stats(result)
    fractions = compute_regime_fractions(result)
    qs_frac = compute_quasi_stable_fraction(result)

    return {
        "scenario": scenario_name,
        "triaxiality": triaxiality,
        "tidal_q": tidal_q,
        "n_orbits": n_orbits,
        "fractions": fractions,
        "stats": stats,
        "quasi_stable_fraction": qs_frac,
        "tidal_epsilon": spin["tidal_epsilon"],
        "e_range": [float(nbody["e"].min()), float(nbody["e"].max())],
        "duration_yr": float((nbody["t"][-1] - nbody["t"][0]) / YEAR),
        "spin_result": spin,
        "regime_result": result,
    }


def _run_worker(job: dict) -> dict:
    """Worker function for parallel execution.

    Thin wrapper around run_single that unpacks a job dict.
    Must be a top-level function for pickling.

    Args:
        job: Dict with keys matching run_single parameters.

    Returns:
        Result dict from run_single, with 'label' and 'elapsed' added.
    """
    t0 = time.time()
    result = run_single(
        scenario_name=job["scenario_name"],
        triaxiality=job["triaxiality"],
        tidal_q=job["tidal_q"],
        n_orbits=job["n_orbits"],
        cli_overrides=job.get("cli_overrides"),
        quiet=True,
    )
    result["elapsed"] = time.time() - t0
    result["label"] = job["label"]
    result["a_au"] = job["a_au"]
    return result


def write_sweep_report(
    all_results: list[dict],
    output_dir: str,
) -> str:
    """Write markdown summary of sweep results.

    Args:
        all_results: List of result dicts from run_single.
        output_dir: Directory to write report.

    Returns:
        Path to report file.
    """
    report_path = os.path.join(output_dir, "tlsr_sweep_report.md")

    lines = ["# TLSR Spin Dynamics Sweep Report\n"]
    lines.append(f"Runs: {len(all_results)}\n")

    # Summary table
    lines.append("| Scenario | (B-A)/C | Q | a (AU) | τ_tide (yr) | Duration (yr) | Ratio | TL_ZERO% | TL_PI% | SPIN% | PTB% |")
    lines.append("|----------|---------|---|--------|-------------|---------------|-------|----------|--------|-------|------|")

    for r in all_results:
        f = r["fractions"]
        a_au = r.get("a_au", BIPOLARIS_DISTANCE_AU)

        # Compute τ_tide for this result
        tau_tide = compute_tidal_timescale(
            STAR_MASS,  # will use overridden value if stored in result
            a_au,
            BIPOLARIS_MASS,
            BIPOLARIS_RADIUS,
            r["tidal_q"],
        )
        ratio = r["duration_yr"] / tau_tide

        lines.append(
            f"| {r['scenario']} | {r['triaxiality']:.0e} | {r['tidal_q']} "
            f"| {a_au:.3f} | {tau_tide:.0f} | {r['duration_yr']:.0f} "
            f"| {ratio:.1f}× "
            f"| {f.get('TL_ZERO', 0) * 100:.1f} "
            f"| {f.get('TL_PI', 0) * 100:.1f} "
            f"| {f.get('SPINNING', 0) * 100:.1f} "
            f"| {f.get('PTB', 0) * 100:.1f} |"
        )

    lines.append("")

    # Per-run details
    for r in all_results:
        label = f"{r['scenario']}_triax{r['triaxiality']:.0e}_Q{r['tidal_q']}"
        lines.append(f"## {label}\n")
        lines.append(f"- Eccentricity range: [{r['e_range'][0]:.6f}, {r['e_range'][1]:.6f}]")
        lines.append(f"- Tidal ε: {r['tidal_epsilon']:.4e} 1/s")
        lines.append(f"- Quasi-stable fraction: {r['quasi_stable_fraction']:.2%}")
        lines.append("")

        for rtype, stats in r["stats"].items():
            if stats["count"] > 0:
                lines.append(
                    f"  - {rtype}: {stats['count']:.0f} regimes, "
                    f"mean={stats['mean_yr']:.1f} yr, "
                    f"median={stats['median_yr']:.1f} yr"
                )
        lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {report_path}")
    return report_path


def compute_tidal_timescale(
    m_star: float,
    a_au: float,
    m_planet: float,
    r_planet: float,
    tidal_q: float,
) -> float:
    """Compute tidal dissipation timescale in years.

    Uses formula from physics.py: ε ∝ (M_star² R³) / (Q Ω m a⁶)
    Returns τ_tide = 1/ε in years.

    Args:
        m_star: Stellar mass (kg).
        a_au: Semi-major axis (AU).
        m_planet: Planet mass (kg).
        r_planet: Planet radius (m).
        tidal_q: Tidal quality factor.

    Returns:
        Tidal timescale in years.
    """
    from shared.constants import G, YEAR

    a_m = a_au * AU  # AU to meters
    n = np.sqrt(G * m_star / a_m**3)  # mean motion
    omega = n  # assume synchronous rotation

    # Tidal dissipation rate: ε = k × (M_star/m_planet) × (R/a)^6 × M_star × R^2 × σ
    # where σ = G / (2 Q Ω R^5)
    sigma = G / (2.0 * tidal_q * omega * r_planet**5)
    eps = (15.0 / 2.0) * (m_star / m_planet) * (r_planet / a_m)**6 * \
          m_star * r_planet**2 * sigma

    return (1.0 / eps) / YEAR


def _report_result(result: dict) -> None:
    """Print timescale summary for a completed run.

    Args:
        result: Result dict from _run_worker.
    """
    a_au = result["a_au"]
    tau_tide = compute_tidal_timescale(
        STAR_MASS, a_au, BIPOLARIS_MASS, BIPOLARIS_RADIUS, result["tidal_q"],
    )
    ratio = result["duration_yr"] / tau_tide
    f = result["fractions"]
    print(f"  Done in {result['elapsed']:.1f}s | "
          f"τ_tide={tau_tide:.0f} yr, {ratio:.1f}× | "
          f"TL0={f.get('TL_ZERO', 0) * 100:.0f}% "
          f"TLπ={f.get('TL_PI', 0) * 100:.0f}% "
          f"SPIN={f.get('SPINNING', 0) * 100:.0f}%")


def _save_run_output(result: dict, out_dir: str) -> None:
    """Save per-run plots and JSON results.

    Args:
        result: Result dict from _run_worker (must contain spin_result, regime_result).
        out_dir: Base output directory.
    """
    label = result["label"]
    run_dir = os.path.join(out_dir, label)
    os.makedirs(run_dir, exist_ok=True)

    plot_spin_history(
        result["spin_result"]["t"],
        result["spin_result"]["gamma"],
        result["regime_result"],
        os.path.join(run_dir, "spin_history.png"),
        title=label,
    )
    plot_regime_pie(
        result["fractions"],
        os.path.join(run_dir, "regime_pie.png"),
        title=f"Regime Fractions — {label}",
    )

    save_result = {k: v for k, v in result.items()
                   if k not in ("spin_result", "regime_result")}
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(save_result, f, indent=2, default=str)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="TLSR spin dynamics sweep")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 10k orbits, 1 scenario, 1 Q, 1 triax")
    parser.add_argument("--scenarios", nargs="*", default=None,
                        help="Scenario names to sweep (default: all)")
    parser.add_argument("--triax", nargs="*", type=float, default=None,
                        help="Triaxiality values to sweep")
    parser.add_argument("--q-values", nargs="*", type=int, default=None,
                        help="Tidal Q values to sweep")
    parser.add_argument("--a-override", type=float, default=None,
                        help="Override BIPOLARIS_DISTANCE_AU (AU)")
    parser.add_argument("--n-orbits-override", type=int, default=None,
                        help="Override n_orbits (for short runs)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Max parallel workers (default: 1, sequential)")
    args = parser.parse_args()

    # Sweep parameters
    if args.quick:
        n_orbits = 10_000
        scenarios = [list(SCENARIOS.keys())[0]]  # just A
        triaxialities = [1e-5]
        tidal_qs = [10]
    else:
        n_orbits = 10_000_000
        scenarios = args.scenarios or list(SCENARIOS.keys())
        triaxialities = args.triax or DEFAULT_TRIAXIALITIES
        tidal_qs = args.q_values or DEFAULT_TIDAL_QS

    # Apply CLI override for n_orbits
    if args.n_orbits_override:
        n_orbits = args.n_orbits_override

    n_total = len(scenarios) * len(triaxialities) * len(tidal_qs)
    n_workers = min(args.workers, n_total)
    print(f"TLSR Sweep: {n_total} runs "
          f"({len(scenarios)} scenarios × {len(triaxialities)} triax × {len(tidal_qs)} Q)")
    print(f"  N-body: {n_orbits:,} orbits per run")
    print(f"  Workers: {n_workers}")

    out_dir = output_dir_for(__file__)

    # Build CLI overrides dict
    cli_overrides: dict = {}
    if args.a_override is not None:
        cli_overrides["BIPOLARIS_DISTANCE_AU"] = args.a_override
        print(f"  CLI override: a = {args.a_override} AU")

    # Build job list
    jobs: list[dict] = []
    for scenario_name in scenarios:
        overrides = get_scenario(scenario_name)
        if cli_overrides:
            overrides.update(cli_overrides)
        a_au = overrides.get("BIPOLARIS_DISTANCE_AU", BIPOLARIS_DISTANCE_AU)

        for triax in triaxialities:
            for q in tidal_qs:
                label = f"{scenario_name}_triax{triax:.0e}_Q{q}"
                jobs.append({
                    "scenario_name": scenario_name,
                    "triaxiality": triax,
                    "tidal_q": q,
                    "n_orbits": n_orbits,
                    "cli_overrides": cli_overrides or None,
                    "label": label,
                    "a_au": a_au,
                })

    # Execute sweep
    all_results: list[dict] = []
    t_sweep_start = time.time()

    if n_workers <= 1:
        # Sequential execution
        for i, job in enumerate(jobs):
            print(f"\n[{i + 1}/{n_total}] {job['label']}")
            result = _run_worker(job)
            _report_result(result)
            _save_run_output(result, out_dir)
            all_results.append(result)
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_job = {
                executor.submit(_run_worker, job): job
                for job in jobs
            }
            for i, future in enumerate(as_completed(future_to_job)):
                result = future.result()
                print(f"\n[{i + 1}/{n_total}] {result['label']}")
                _report_result(result)
                _save_run_output(result, out_dir)
                all_results.append(result)

    # Sort results back to deterministic order (scenario, triax, Q)
    all_results.sort(key=lambda r: (r["scenario"], r["triaxiality"], r["tidal_q"]))

    elapsed_total = time.time() - t_sweep_start
    print(f"\nAll {n_total} runs completed in {elapsed_total:.1f}s")

    # Summary
    write_sweep_report(all_results, out_dir)

    # Comparison figure
    summary_data = {
        f"{r['scenario']}_triax{r['triaxiality']:.0e}_Q{r['tidal_q']}": {
            "fractions": r["fractions"],
            "quasi_stable_fraction": r["quasi_stable_fraction"],
        }
        for r in all_results
    }
    plot_regime_summary(
        summary_data,
        os.path.join(out_dir, "regime_comparison.png"),
    )

    print(f"\nSweep complete. Results in: {out_dir}")


if __name__ == "__main__":
    main()
