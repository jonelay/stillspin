#!/usr/bin/env python3
"""Bipolaris v3.6 Sensitivity and Stability Analysis.

Quantifies parameter sensitivity and rarity of the balanced-regime configuration
to assess credibility for hard sci-fi worldbuilding.

Twelve studies:
1. Q-Distance Grid: Map viable parameter space (50 configs)
2. Monte Carlo Rarity: Estimate probability with credibility-weighted priors (2000 configs)
3. Resonance Robustness: Characterize distance sensitivity (21 configs)
4. Q Boundary Detection: Find minimum/maximum viable Q (39 configs)
5. Slow Bouncer Discovery: Find worlds with long TL, brief PTB (352 configs)
6. Thermal Optimization: Optimize albedo/CO2 for top candidates (500 configs)
7. Rarity Estimation: Monte Carlo with slow bouncer priors (2000 configs)
8. Low-Q Exploration: Test Q=8-14 for slow bouncers (108 configs)
9. Fine Distance Resolution: Map bifurcation at 100 μAU (84 configs)
10. Stellar Mass Sweep: M-dwarf mass exploration (72 configs)
11. Low-Q Flip-Flop Search: Find flip-flop worlds at outer distances (60 configs)
12. Bifurcation Mapping: Fine-resolution mapping at d=0.0745 AU (132 configs)

Total: ~5416 configurations.

Infrastructure v3 (2026-02-12):
- Crash-safe incremental saving via JSONL
- Timeout protection (600s default, 900s for risky configs)
- Resume support via config deduplication
- Validated SweepConfig dataclass
- Episode extraction for temporal pattern analysis
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.analysis import (
    classify_config_risk,
    compute_episode_statistics,
    compute_slow_bouncer_score,
    compute_surface_conditions,
    estimate_rarity_probability,
    filter_slow_bouncer_candidates,
    fit_resonance_profile,
)
from shared.result_store import ResultStore
from shared.sweep_runner import run_config_safe
from shared.sweep_types import SweepConfig, SweepResult

# v3.2 baseline TLSR parameters
BASE_TRIAX = 3e-5
BASE_Q = 20
BASE_ALBEDO = 0.35
BASE_CO2_PCT = 0.6  # 0.6% = 0.006 fraction

# Timeout settings (seconds)
TIMEOUT_DEFAULT = 600  # 10 min for normal configs
TIMEOUT_RISKY = 900    # 15 min for risky configs


def _run_config_with_risk_timeout(args: tuple) -> "SweepResult":
    """Module-level wrapper for ProcessPoolExecutor (must be picklable)."""
    config, default_timeout = args
    risk = classify_config_risk(config)
    cfg_timeout = TIMEOUT_RISKY if risk == "risky" else default_timeout
    return run_config_safe(config, timeout_s=cfg_timeout)


def _result_to_legacy_format(result) -> dict:
    """Convert SweepResult to legacy dict format for backwards compatibility."""
    base = {
        "config": result.config.to_dict(),
        "elapsed": result.elapsed_s,
    }
    if result.status != "OK":
        base["status"] = result.status
        base["error_msg"] = result.error_msg
    else:
        base["temps"] = result.temps
        base["fractions"] = result.fractions
        base["quality"] = result.quality
    return base


def _execute_parallel(
    configs: list[SweepConfig],
    store: ResultStore,
    n_workers: int,
    label: str,
    timeout_s: int = TIMEOUT_DEFAULT,
) -> list[dict]:
    """Execute configs in parallel with progress, resume, and incremental saving."""
    completed_ids = store.get_completed_ids()
    pending = [c for c in configs if c.config_id not in completed_ids]

    n_completed = len(completed_ids)
    n_total = len(configs)
    n_pending = len(pending)

    if n_completed > 0:
        print(f"Resuming: {n_completed} already completed, {n_pending} remaining")

    if n_pending == 0:
        print("All configs already completed")
        return [_result_to_legacy_format(r) for r in store.load_results()]

    # Sort by risk: safe first, risky last
    pending.sort(key=lambda c: {"safe": 0, "moderate": 1, "risky": 2}[classify_config_risk(c)])

    n_workers = min(n_workers, n_pending)
    print(f"Workers: {n_workers}, Timeout: {timeout_s}s")

    t_start = time.time()

    # run_config_safe now handles timeout internally via subprocess.run(timeout=X)
    # which actually kills hung processes (unlike ProcessPoolExecutor.future.result)

    if n_workers <= 1:
        for i, config in enumerate(pending):
            result = _run_config_with_risk_timeout((config, timeout_s))
            store.save(result)
            _print_result(result, i + 1, n_pending)
    else:
        # Prepare args tuples for pickling (can't use closures with ProcessPoolExecutor)
        config_args = [(config, timeout_s) for config in pending]

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_run_config_with_risk_timeout, args): args[0]
                       for args in config_args}

            for i, future in enumerate(as_completed(futures)):
                config = futures[future]
                try:
                    result = future.result()  # No timeout here - handled by subprocess
                except Exception as e:
                    result = SweepResult(
                        config=config,
                        status="ERROR",
                        fractions=None,
                        temps=None,
                        quality=None,
                        elapsed_s=0,
                        error_msg=str(e),
                    )
                store.save(result)
                _print_result(result, i + 1, n_pending)

    elapsed = time.time() - t_start
    stats = store.get_stats()
    print(f"{label} completed in {elapsed:.1f}s ({elapsed/max(1, n_pending):.1f}s per config)")
    print(f"  OK: {stats['ok']}, TIMEOUT: {stats['timeout']}, ERROR: {stats['error']}")

    return [_result_to_legacy_format(r) for r in store.load_results()]


def _print_result(result, idx: int, total: int):
    """Print single result summary."""
    cfg = result.config
    if result.status != "OK":
        print(f"[{idx}/{total}] Q={cfg.q}, dist={cfg.distance_au:.4f} AU → "
              f"{result.status}: {result.error_msg or 'unknown'}")
        return
    f = result.fractions
    print(f"[{idx}/{total}] Q={cfg.q}, dist={cfg.distance_au:.4f} AU → "
          f"PTB={f.get('PTB', 0)*100:.1f}%, score={result.quality['overall_score']:.3f}, "
          f"T_term={result.temps['t_term']}K")


def rerun_specific_configs(args, output_dir: Path) -> None:
    """Re-run specific configs identified by ID."""
    print("\n=== Re-running Specific Configs ===")

    if not args.resume_configs.exists():
        print(f"Error: Config file not found: {args.resume_configs}")
        sys.exit(1)

    # Load config IDs to re-run
    with open(args.resume_configs) as f:
        config_ids = {line.strip() for line in f if line.strip()}

    print(f"Loaded {len(config_ids)} config IDs from {args.resume_configs}")

    # Find configs across all studies
    all_configs = []
    for study_name in ["study1", "study2", "study3", "study4", "study5", "study6", "study7"]:
        study_dir = output_dir / study_name
        if not study_dir.exists():
            continue

        store = ResultStore(study_dir)
        results = store.load_results()

        for result in results:
            if result.config.config_id in config_ids:
                all_configs.append((study_name, result.config))

    if not all_configs:
        print("No matching configs found in existing results")
        return

    print(f"Found {len(all_configs)} configs to re-run")

    # Group by study and re-run
    by_study = {}
    for study_name, config in all_configs:
        by_study.setdefault(study_name, []).append(config)

    for study_name, configs in sorted(by_study.items()):
        print(f"\n{study_name}: {len(configs)} configs")
        store = ResultStore(output_dir / study_name)

        # Mark configs as pending to force re-run (removes from index)
        config_ids_to_rerun = {c.config_id for c in configs}
        store.mark_pending(config_ids_to_rerun)

        # Re-run with episode extraction enabled (results append to JSONL)
        _execute_parallel(configs, store, args.workers, f"{study_name} re-run", args.timeout)

    print("\n✓ Re-run complete")


def study1_q_distance_grid(args, output_dir: Path) -> list[dict]:
    """Study 1: Q-Distance Parameter Space Mapping."""
    print("\n=== Study 1: Q-Distance Grid ===")

    store = ResultStore(output_dir / "study1")

    q_values = [10, 12, 15, 18, 20, 22, 25, 30, 35, 40]
    distances = [0.068, 0.069, 0.070, 0.071, 0.072]

    configs = []
    for q in q_values:
        for dist in distances:
            configs.append(SweepConfig(
                q=q,
                distance_au=dist,
                triax=BASE_TRIAX,
                albedo=BASE_ALBEDO,
                co2_pct=BASE_CO2_PCT,
                n_orbits=args.n_orbits,
            ))

    print(f"Configurations: {len(configs)} (Q: {len(q_values)}, distance: {len(distances)})")

    results = _execute_parallel(configs, store, args.workers, "Study 1", args.timeout)

    # Also save legacy JSON format
    output_file = output_dir / "study1_q_distance_grid.json"
    ok_results = [r for r in results if r.get("status", "OK") == "OK" or "fractions" in r]
    with open(output_file, "w") as f:
        json.dump(ok_results, f, indent=2, default=str)
    print(f"Saved: {output_file}")

    return ok_results


def study2_monte_carlo_rarity(args, output_dir: Path) -> tuple[list[dict], dict]:
    """Study 2: Monte Carlo Rarity Estimation.

    v3.3 adjustments: Center at 0.0715 AU (between resonance peaks), wider sigma.
    """
    print("\n=== Study 2: Monte Carlo Rarity Estimation ===")

    store = ResultStore(output_dir / "study2")

    n_samples = 2000 if not args.quick else 100
    print(f"Samples: {n_samples}")

    np.random.seed(42)
    q_samples = np.exp(np.random.uniform(np.log(5), np.log(50), n_samples))
    # v3.3: Center between resonance peaks at 0.0715 AU, wider spread
    dist_samples = np.random.normal(0.0715, 0.003, n_samples)
    triax_samples = np.random.uniform(2.5e-5, 3.5e-5, n_samples)
    albedo_samples = np.random.uniform(0.30, 0.40, n_samples)
    co2_samples = np.random.uniform(0.4, 0.8, n_samples)

    configs = []
    for i in range(n_samples):
        try:
            configs.append(SweepConfig(
                q=int(q_samples[i]),
                distance_au=float(np.clip(dist_samples[i], 0.04, 0.10)),
                triax=float(triax_samples[i]),
                albedo=float(albedo_samples[i]),
                co2_pct=float(co2_samples[i]),
                n_orbits=args.n_orbits,
                seed=i,
            ))
        except ValueError:
            continue

    print(f"Valid configs after filtering: {len(configs)}")

    results = _execute_parallel(configs, store, args.workers, "Study 2", args.timeout)

    # Compute rarity from OK results only
    ok_results = [r for r in results if r.get("quality")]
    if ok_results:
        scores = [r["quality"]["overall_score"] for r in ok_results]
        rarity_stats = estimate_rarity_probability(scores, threshold=0.5, method="kde")
    else:
        rarity_stats = {"p_viable": 0, "rarity": float("inf"), "n_viable": 0, "n_total": 0,
                        "ci_lower": 0, "ci_upper": 0}

    output_file = output_dir / "study2_monte_carlo.json"
    with open(output_file, "w") as f:
        json.dump({"results": ok_results, "rarity_stats": rarity_stats}, f, indent=2, default=str)
    print(f"Saved: {output_file}")

    if rarity_stats["p_viable"] > 0:
        print(f"\nRarity estimate: 1:{rarity_stats['rarity']:.0f}")
        print(f"  P(viable) = {rarity_stats['p_viable']:.4f} "
              f"[{rarity_stats['ci_lower']:.4f}, {rarity_stats['ci_upper']:.4f}]")
    print(f"  Viable configs: {rarity_stats['n_viable']}/{rarity_stats['n_total']}")

    return ok_results, rarity_stats


def study3_resonance_robustness(args, output_dir: Path) -> tuple[list[dict], dict, dict]:
    """Study 3: Resonance Robustness Analysis."""
    print("\n=== Study 3: Resonance Robustness ===")

    store = ResultStore(output_dir / "study3")

    distances = np.arange(0.065, 0.0755, 0.0005)

    configs = []
    for dist in distances:
        configs.append(SweepConfig(
            q=BASE_Q,
            distance_au=float(dist),
            triax=BASE_TRIAX,
            albedo=BASE_ALBEDO,
            co2_pct=BASE_CO2_PCT,
            n_orbits=args.n_orbits,
        ))

    print(f"Configurations: {len(configs)} (distance range: {distances[0]:.3f} - {distances[-1]:.3f} AU)")

    results = _execute_parallel(configs, store, args.workers, "Study 3", args.timeout)

    ok_results = [r for r in results if r.get("fractions")]
    if ok_results:
        dists = np.array([r["config"]["distance_au"] for r in ok_results])
        ptb_fracs = np.array([r["fractions"].get("PTB", 0) for r in ok_results])
        fit_gaussian = fit_resonance_profile(dists, ptb_fracs, model="gaussian")
        fit_lorentzian = fit_resonance_profile(dists, ptb_fracs, model="lorentzian")
    else:
        fit_gaussian = fit_lorentzian = {"fwhm": float("nan"), "center": float("nan"), "r_squared": 0}

    output_file = output_dir / "study3_resonance_profile.json"
    with open(output_file, "w") as f:
        json.dump({
            "results": ok_results,
            "fit_gaussian": {k: v for k, v in fit_gaussian.items() if k != "func"},
            "fit_lorentzian": {k: v for k, v in fit_lorentzian.items() if k != "func"},
        }, f, indent=2, default=str)
    print(f"Saved: {output_file}")

    if not np.isnan(fit_gaussian.get("fwhm", float("nan"))):
        print(f"\nResonance window (Gaussian): FWHM = {fit_gaussian['fwhm']*1000:.2f} mAU, "
              f"center = {fit_gaussian['center']:.4f} AU (R² = {fit_gaussian['r_squared']:.3f})")
        print(f"Resonance window (Lorentzian): FWHM = {fit_lorentzian['fwhm']*1000:.2f} mAU, "
              f"center = {fit_lorentzian['center']:.4f} AU (R² = {fit_lorentzian['r_squared']:.3f})")

    return ok_results, fit_gaussian, fit_lorentzian


def study4_q_boundary_detection(args, output_dir: Path) -> tuple[list[dict], int | None, int | None]:
    """Study 4: Extended Q-Range Boundary Detection."""
    print("\n=== Study 4: Q Boundary Detection ===")

    store = ResultStore(output_dir / "study4")

    q_values = [5, 7, 10, 12, 15, 18, 20, 25, 30, 40, 50, 75, 100]
    triax_values = [2.8e-5, 3.0e-5, 3.2e-5]

    configs = []
    for q in q_values:
        for triax in triax_values:
            configs.append(SweepConfig(
                q=q,
                distance_au=0.070,
                triax=triax,
                albedo=BASE_ALBEDO,
                co2_pct=BASE_CO2_PCT,
                n_orbits=args.n_orbits,
            ))

    print(f"Configurations: {len(configs)} (Q: {len(q_values)}, triax: {len(triax_values)})")

    results = _execute_parallel(configs, store, args.workers, "Study 4", args.timeout)

    ok_results = [r for r in results if r.get("quality")]
    viable_qs = [r["config"]["q"] for r in ok_results if r["quality"].get("is_viable")]
    if viable_qs:
        q_min = min(viable_qs)
        q_max = max(viable_qs)
        print(f"\nViable Q range: [{q_min}, {q_max}]")
    else:
        print("\nNo viable configurations found!")
        q_min = q_max = None

    output_file = output_dir / "study4_q_boundary.json"
    with open(output_file, "w") as f:
        json.dump({"results": ok_results, "q_min": q_min, "q_max": q_max}, f, indent=2, default=str)
    print(f"Saved: {output_file}")

    return ok_results, q_min, q_max


def study5_slow_bouncer_discovery(args, output_dir: Path) -> tuple[list[dict], list[dict]]:
    """Study 5: Slow Bouncer Discovery Sweep.

    Searches resonance valleys (not peaks) with moderate Q to find worlds with:
    - Long stable TL periods (~50-200 years)
    - Brief PTB transitions (<30% time)
    - <50 episodes per 1000 years
    """
    print("\n=== Study 5: Slow Bouncer Discovery ===")

    store = ResultStore(output_dir / "study5")

    # Q range focused on moderate values (avoids chaotic low-Q and stable high-Q)
    q_values = [15, 18, 20, 22, 25, 28, 30, 35]

    # Resonance valleys (between peaks at 0.0675, 0.0695, 0.072, 0.0735)
    distances = [
        0.0680, 0.0685, 0.0690,  # Valley between 0.0675 and 0.0695
        0.0710, 0.0715, 0.0720, 0.0725,  # Valley between 0.0695 and 0.072
        0.0745, 0.0750, 0.0755, 0.0760,  # Valley beyond 0.0735
    ]

    # Triaxiality sweep (low values for slower dynamics)
    triax_values = [2e-5, 3e-5, 4e-5, 5e-5]

    configs = []
    for q in q_values:
        for dist in distances:
            for triax in triax_values:
                configs.append(SweepConfig(
                    q=q,
                    distance_au=dist,
                    triax=triax,
                    albedo=0.35,
                    co2_pct=0.6,
                    n_orbits=args.n_orbits,
                ))

    if args.quick:
        configs = configs[:40]

    print(f"Configurations: {len(configs)} (Q: {len(q_values)}, distance: {len(distances)}, triax: {len(triax_values)})")

    results = _execute_parallel(configs, store, args.workers, "Study 5", args.timeout)

    # Filter to slow bouncer candidates
    ok_results = store.load_results()
    candidates = filter_slow_bouncer_candidates(ok_results)

    # Save results
    output_file = output_dir / "study5_slow_bouncer_grid.json"
    ok_dicts = [r for r in results if r.get("status", "OK") == "OK" or "fractions" in r]
    with open(output_file, "w") as f:
        json.dump(ok_dicts, f, indent=2, default=str)

    candidates_file = output_dir / "study5_candidates.json"
    with open(candidates_file, "w") as f:
        json.dump(candidates, f, indent=2, default=str)

    print(f"Saved: {output_file}")
    print(f"Saved: {candidates_file}")
    print(f"\nSlow bouncer candidates: {len(candidates)}")

    if candidates:
        top = candidates[0]
        print(f"Best candidate: Q={top['config']['q']}, dist={top['config']['distance_au']:.4f} AU, "
              f"score={top['score_details']['slow_bouncer_score']:.3f}")

    return ok_dicts, candidates


def study6_thermal_optimization(
    args, output_dir: Path, candidates: list[dict] | None = None
) -> list[dict]:
    """Study 6: Thermal Optimization for Top Candidates.

    Takes top 20 slow bouncer candidates from Study 5 and sweeps
    albedo/CO2 to find optimal thermal conditions.
    """
    print("\n=== Study 6: Thermal Optimization ===")

    # Load candidates from file or use provided list
    if candidates is None:
        candidates_file = args.study5_candidates
        if candidates_file is None:
            candidates_file = output_dir / "study5_candidates.json"

        if not candidates_file.exists():
            print(f"Error: No candidates file found at {candidates_file}")
            print("Run Study 5 first or provide --study5-candidates path")
            return []

        with open(candidates_file) as f:
            candidates = json.load(f)

    if not candidates:
        print("No slow bouncer candidates to optimize")
        return []

    # Take top 20 candidates
    top_candidates = candidates[:20]
    print(f"Optimizing top {len(top_candidates)} candidates")

    store = ResultStore(output_dir / "study6")

    # Thermal parameter grid
    albedo_values = [0.30, 0.35, 0.40, 0.45, 0.50]
    co2_values = [0.2, 0.4, 0.6, 0.8, 1.0]  # percent

    configs = []
    for cand in top_candidates:
        base_config = cand["config"]
        for albedo in albedo_values:
            for co2 in co2_values:
                configs.append(SweepConfig(
                    q=base_config["q"],
                    distance_au=base_config["distance_au"],
                    triax=base_config["triax"],
                    albedo=albedo,
                    co2_pct=co2,
                    n_orbits=args.n_orbits,
                ))

    if args.quick:
        configs = configs[:50]

    print(f"Configurations: {len(configs)} ({len(top_candidates)} candidates × "
          f"{len(albedo_values)} albedo × {len(co2_values)} CO2)")

    results = _execute_parallel(configs, store, args.workers, "Study 6", args.timeout)

    # Find best thermal configs
    ok_results = store.load_results()
    optimized = filter_slow_bouncer_candidates(ok_results)

    output_file = output_dir / "study6_thermal_optimization.json"
    with open(output_file, "w") as f:
        json.dump(optimized, f, indent=2, default=str)
    print(f"Saved: {output_file}")

    if optimized:
        best = optimized[0]
        print(f"\nBest thermal config: albedo={best['config']['albedo']}, "
              f"CO2={best['config']['co2_pct']}%, T_term={best['temps'].get('t_term', 'N/A')}K")

    return optimized


def study7_rarity_estimation(args, output_dir: Path) -> tuple[list[dict], dict]:
    """Study 7: Monte Carlo Rarity Estimation with Slow Bouncer Priors.

    Uses physically motivated priors centered on slow bouncer parameter regions.
    """
    print("\n=== Study 7: Slow Bouncer Rarity Estimation ===")

    store = ResultStore(output_dir / "study7")

    n_samples = 2000 if not args.quick else 100
    print(f"Samples: {n_samples}")

    np.random.seed(42)

    # Q: Log-uniform [15, 35] (moderate range for slow dynamics)
    q_samples = np.exp(np.random.uniform(np.log(15), np.log(35), n_samples))

    # Distance: Gaussian mixture at valleys (0.0685, 0.0715 AU)
    valley_choice = np.random.choice([0.0685, 0.0715], n_samples)
    dist_samples = np.random.normal(valley_choice, 0.0015)

    # Triaxiality: Uniform [2e-5, 5e-5]
    triax_samples = np.random.uniform(2e-5, 5e-5, n_samples)

    # Albedo: Uniform [0.30, 0.45]
    albedo_samples = np.random.uniform(0.30, 0.45, n_samples)

    # CO2: Uniform [0.3, 0.8]%
    co2_samples = np.random.uniform(0.3, 0.8, n_samples)

    configs = []
    for i in range(n_samples):
        try:
            configs.append(SweepConfig(
                q=int(q_samples[i]),
                distance_au=float(np.clip(dist_samples[i], 0.04, 0.10)),
                triax=float(triax_samples[i]),
                albedo=float(albedo_samples[i]),
                co2_pct=float(co2_samples[i]),
                n_orbits=args.n_orbits,
                seed=i,
            ))
        except ValueError:
            continue

    print(f"Valid configs after filtering: {len(configs)}")

    results = _execute_parallel(configs, store, args.workers, "Study 7", args.timeout)

    # Filter to slow bouncers and compute rarity
    ok_results = store.load_results()
    candidates = filter_slow_bouncer_candidates(ok_results)

    n_slow_bouncer = len(candidates)
    n_total = len(ok_results)

    # Compute rarity
    if n_total > 0:
        p_viable = n_slow_bouncer / n_total
        # Wilson score interval
        z = 1.96
        if n_slow_bouncer > 0:
            denom = 1 + z**2 / n_total
            center = (p_viable + z**2 / (2 * n_total)) / denom
            margin = z * np.sqrt(p_viable * (1 - p_viable) / n_total + z**2 / (4 * n_total**2)) / denom
            ci_lower = max(0, center - margin)
            ci_upper = min(1, center + margin)
        else:
            ci_lower = ci_upper = 0.0
        rarity = 1 / p_viable if p_viable > 0 else float("inf")
    else:
        p_viable = 0.0
        ci_lower = ci_upper = 0.0
        rarity = float("inf")

    rarity_stats = {
        "p_viable": p_viable,
        "rarity": rarity,
        "n_viable": n_slow_bouncer,
        "n_total": n_total,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }

    output_file = output_dir / "study7_rarity.json"
    with open(output_file, "w") as f:
        json.dump({
            "rarity_stats": rarity_stats,
            "candidates": candidates[:50],  # Top 50
        }, f, indent=2, default=str)
    print(f"Saved: {output_file}")

    print(f"\nSlow bouncer rarity: 1:{rarity:.0f}")
    print(f"  P(slow_bouncer) = {p_viable:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  Found: {n_slow_bouncer}/{n_total}")

    ok_dicts = [_result_to_legacy_format(r) for r in ok_results if r.status == "OK"]
    return ok_dicts, rarity_stats


def study8_low_q_sweep(args, output_dir: Path) -> list[dict]:
    """Study 8: Low-Q Parameter Exploration.

    Tests hypothesis that Q=8-14 produces slow bouncers near the bifurcation
    boundary (0.072-0.076 AU). Lower Q means stronger tidal dissipation,
    potentially enabling longer stable TL episodes with brief PTB transitions.

    Based on Study 5 finding that parameter space bifurcates at ~0.072 AU.
    """
    print("\n=== Study 8: Low-Q Parameter Exploration ===")

    store = ResultStore(output_dir / "study8")

    # Q range below Study 5 minimum (15)
    q_values = [8, 10, 12, 14]

    # Distance grid spanning bifurcation boundary
    distances = [0.0720, 0.0725, 0.0730, 0.0735, 0.0740, 0.0745, 0.0750, 0.0755, 0.0760]

    # Triaxiality values
    triax_values = [2e-5, 3e-5, 4e-5]

    configs = []
    for q in q_values:
        for dist in distances:
            for triax in triax_values:
                configs.append(SweepConfig(
                    q=q,
                    distance_au=dist,
                    triax=triax,
                    albedo=BASE_ALBEDO,
                    co2_pct=BASE_CO2_PCT,
                    n_orbits=args.n_orbits,
                ))

    if args.quick:
        configs = configs[:20]

    print(f"Configurations: {len(configs)} (Q: {len(q_values)}, distance: {len(distances)}, triax: {len(triax_values)})")

    results = _execute_parallel(configs, store, args.workers, "Study 8", args.timeout)

    # Filter to slow bouncer candidates
    ok_results = store.load_results()
    candidates = filter_slow_bouncer_candidates(ok_results)

    output_file = output_dir / "study8_low_q_sweep.json"
    ok_dicts = [r for r in results if r.get("status", "OK") == "OK" or "fractions" in r]
    with open(output_file, "w") as f:
        json.dump(ok_dicts, f, indent=2, default=str)

    candidates_file = output_dir / "study8_candidates.json"
    with open(candidates_file, "w") as f:
        json.dump(candidates, f, indent=2, default=str)

    print(f"Saved: {output_file}")
    print(f"Saved: {candidates_file}")
    print(f"\nSlow bouncer candidates: {len(candidates)}")

    if candidates:
        top = candidates[0]
        print(f"Best candidate: Q={top['config']['q']}, dist={top['config']['distance_au']:.4f} AU, "
              f"score={top['score_details']['slow_bouncer_score']:.3f}")

    return ok_dicts, candidates


def study9_fine_distance_sweep(args, output_dir: Path) -> list[dict]:
    """Study 9: Fine Distance Resolution Sweep.

    Maps the bifurcation boundary with 100 μAU resolution to precisely
    identify the transition between stable TL_ZERO and flip-flop regimes.

    84 configs (2 Q × 42 distance points).
    """
    print("\n=== Study 9: Fine Distance Resolution Sweep ===")

    store = ResultStore(output_dir / "study9")

    # Fixed Q at v2.2 validated range
    q_values = [20, 22]

    # Fine distance grid: 0.072 to 0.076 AU in 100 μAU steps
    distances = np.arange(0.0720, 0.0760 + 0.0001, 0.0001)

    configs = []
    for q in q_values:
        for dist in distances:
            configs.append(SweepConfig(
                q=q,
                distance_au=float(dist),
                triax=BASE_TRIAX,
                albedo=BASE_ALBEDO,
                co2_pct=BASE_CO2_PCT,
                n_orbits=args.n_orbits,
            ))

    if args.quick:
        # Sample every 10th distance point
        configs = configs[::10]

    print(f"Configurations: {len(configs)} (Q: {len(q_values)}, distance: {len(distances)} @ 100μAU resolution)")

    results = _execute_parallel(configs, store, args.workers, "Study 9", args.timeout)

    output_file = output_dir / "study9_fine_distance.json"
    ok_dicts = [r for r in results if r.get("status", "OK") == "OK" or "fractions" in r]
    with open(output_file, "w") as f:
        json.dump(ok_dicts, f, indent=2, default=str)

    print(f"Saved: {output_file}")

    # Find bifurcation point (where PTB fraction jumps)
    if ok_dicts:
        sorted_by_dist = sorted(ok_dicts, key=lambda r: r["config"]["distance_au"])
        for i in range(1, len(sorted_by_dist)):
            prev_ptb = sorted_by_dist[i-1].get("fractions", {}).get("PTB", 0)
            curr_ptb = sorted_by_dist[i].get("fractions", {}).get("PTB", 0)
            if abs(curr_ptb - prev_ptb) > 0.2:  # 20% jump
                bifurc_dist = sorted_by_dist[i]["config"]["distance_au"]
                print(f"\nBifurcation detected near {bifurc_dist:.4f} AU")
                print(f"  PTB jump: {prev_ptb*100:.1f}% → {curr_ptb*100:.1f}%")
                break

    return ok_dicts


def study11_low_q_flipflop(args, output_dir: Path) -> tuple[list[dict], list[dict]]:
    """Study 11: Low-Q Flip-Flop Search.

    Extends Study 8 to find worlds with actual flip-flop dynamics (not permanent lock).
    Targets outer distances (0.074-0.076 AU) where PTB activity should emerge.

    Based on Study 8 finding that all 12 candidates were 100% TL_ZERO permanent locks.
    """
    print("\n=== Study 11: Low-Q Flip-Flop Search ===")

    store = ResultStore(output_dir / "study11")

    # Low Q values (same as Study 8)
    q_values = [8, 10, 12, 14]

    # Outer distances beyond stable lock boundary (~0.072 AU)
    distances = [0.0740, 0.0745, 0.0750, 0.0755, 0.0760]

    # Triaxiality values
    triax_values = [2e-5, 3e-5, 4e-5]

    configs = []
    for q in q_values:
        for dist in distances:
            for triax in triax_values:
                configs.append(SweepConfig(
                    q=q,
                    distance_au=dist,
                    triax=triax,
                    albedo=BASE_ALBEDO,
                    co2_pct=BASE_CO2_PCT,
                    n_orbits=args.n_orbits,
                ))

    if args.quick:
        configs = configs[:15]

    print(f"Configurations: {len(configs)} (Q: {len(q_values)}, distance: {len(distances)}, triax: {len(triax_values)})")

    results = _execute_parallel(configs, store, args.workers, "Study 11", args.timeout)

    # Filter to flip-flop slow bouncer candidates (with new PTB range filter)
    ok_results = store.load_results()
    candidates = filter_slow_bouncer_candidates(
        ok_results,
        min_tl_frac=0.70,
        min_ptb_frac=0.05,
        max_ptb_frac=0.30,
        require_flipflop=True,
    )

    output_file = output_dir / "study11_low_q_flipflop.json"
    ok_dicts = [r for r in results if r.get("status", "OK") == "OK" or "fractions" in r]
    with open(output_file, "w") as f:
        json.dump(ok_dicts, f, indent=2, default=str)

    candidates_file = output_dir / "study11_candidates.json"
    with open(candidates_file, "w") as f:
        json.dump(candidates, f, indent=2, default=str)

    print(f"Saved: {output_file}")
    print(f"Saved: {candidates_file}")
    print(f"\nFlip-flop slow bouncer candidates: {len(candidates)}")

    if candidates:
        top = candidates[0]
        print(f"Best candidate: Q={top['config']['q']}, dist={top['config']['distance_au']:.4f} AU, "
              f"score={top['score_details']['slow_bouncer_score']:.3f}, "
              f"PTB={top['fractions'].get('PTB', 0)*100:.1f}%")
    else:
        # Show PTB distribution even if no candidates
        print("\nPTB distribution in results:")
        ptb_values = [r.get("fractions", {}).get("PTB", 0) for r in ok_dicts if r.get("fractions")]
        if ptb_values:
            ptb_values = sorted(ptb_values)
            print(f"  Min: {min(ptb_values)*100:.1f}%, Max: {max(ptb_values)*100:.1f}%")
            print(f"  In 5-30% range: {sum(1 for p in ptb_values if 0.05 <= p <= 0.30)}/{len(ptb_values)}")

    return ok_dicts, candidates


def study12_bifurcation_mapping(args, output_dir: Path) -> list[dict]:
    """Study 12: Bifurcation Boundary Mapping.

    Maps the flip-flop bifurcation boundary at d=0.0745 AU with 100 µAU resolution.
    Study 11 found:
    - d=0.074: SPINNING/PTB, no locks
    - d=0.0745: Flip-flops (TL_ZERO ↔ TL_PI)
    - d=0.075: SPINNING, minimal PTB

    Maps 0.0740-0.0750 AU with 100 µAU steps to find exact bifurcation boundaries.

    132 configs (11 distances × 4 Q × 3 triax).
    """
    print("\n=== Study 12: Bifurcation Boundary Mapping ===")

    store = ResultStore(output_dir / "study12")

    # Fine distance grid: 0.0740-0.0750 AU in 100 µAU steps
    distances = np.linspace(0.0740, 0.0750, 11)

    # Q values (same as Study 11)
    q_values = [8, 10, 12, 14]

    # Triaxiality values (same as Study 11)
    triax_values = [2e-5, 3e-5, 4e-5]

    configs = []
    for q in q_values:
        for dist in distances:
            for triax in triax_values:
                configs.append(SweepConfig(
                    q=q,
                    distance_au=float(dist),
                    triax=triax,
                    albedo=BASE_ALBEDO,
                    co2_pct=BASE_CO2_PCT,
                    n_orbits=args.n_orbits,
                ))

    if args.quick:
        configs = configs[:15]

    print(f"Configurations: {len(configs)} (distance: {len(distances)}, Q: {len(q_values)}, triax: {len(triax_values)})")
    print(f"Distance range: {min(distances):.5f}-{max(distances):.5f} AU (100 µAU steps)")

    results = _execute_parallel(configs, store, args.workers, "Study 12", args.timeout)

    # Analyze bifurcation structure
    ok_dicts = [r for r in results if r.get("status", "OK") == "OK" or "fractions" in r]

    # Classify by regime dominance
    flipflop_count = 0
    spinning_count = 0
    locked_count = 0

    for r in ok_dicts:
        fracs = r.get("fractions", {})
        if not fracs:
            continue

        tl0 = fracs.get("TL_ZERO", 0)
        tlpi = fracs.get("TL_PI", 0)
        spin = fracs.get("SPINNING", 0)
        ptb = fracs.get("PTB", 0)

        # Flip-flop: both TL regimes present with PTB in range
        if tl0 > 0.05 and tlpi > 0.05 and 0.05 <= ptb <= 0.30:
            flipflop_count += 1
        # Permanent lock: one TL regime dominates
        elif (tl0 > 0.70 or tlpi > 0.70) and ptb < 0.30:
            locked_count += 1
        # SPINNING dominant
        elif spin > 0.50:
            spinning_count += 1

    output_file = output_dir / "study12_bifurcation_mapping.json"
    with open(output_file, "w") as f:
        json.dump(ok_dicts, f, indent=2, default=str)

    print(f"Saved: {output_file}")
    print(f"\nRegime distribution:")
    print(f"  Flip-flops: {flipflop_count}/{len(ok_dicts)} ({flipflop_count/len(ok_dicts)*100:.1f}%)")
    print(f"  Locked: {locked_count}/{len(ok_dicts)} ({locked_count/len(ok_dicts)*100:.1f}%)")
    print(f"  Spinning: {spinning_count}/{len(ok_dicts)} ({spinning_count/len(ok_dicts)*100:.1f}%)")

    return ok_dicts


def study10_stellar_mass_sweep(args, output_dir: Path) -> list[dict]:
    """Study 10: M-dwarf Stellar Mass Sweep.

    Explores Bipolaris-like worlds around other M-dwarfs by varying stellar
    mass from 0.08-0.20 M_sun (M8V to M4V). HZ distance is auto-scaled
    using L ∝ M^4 relationship.

    72 configs (6 stellar masses × 4 Q × 3 triax).
    """
    print("\n=== Study 10: M-dwarf Stellar Mass Sweep ===")

    store = ResultStore(output_dir / "study10")

    # Stellar masses: M8V to M4V
    stellar_masses = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20]

    # Q values spanning validated range
    q_values = [18, 20, 22, 25]

    # Triaxiality values
    triax_values = [2e-5, 3e-5, 4e-5]

    # Import HZ calculator
    from shared.analysis import hz_distance_for_mass

    configs = []
    for m_star in stellar_masses:
        # Get HZ center for this stellar mass
        _, hz_center, _ = hz_distance_for_mass(m_star)

        for q in q_values:
            for triax in triax_values:
                configs.append(SweepConfig(
                    q=q,
                    distance_au=hz_center,
                    triax=triax,
                    albedo=BASE_ALBEDO,
                    co2_pct=BASE_CO2_PCT,
                    n_orbits=args.n_orbits,
                    stellar_mass_msun=m_star,
                ))

    if args.quick:
        configs = configs[:15]

    print(f"Configurations: {len(configs)} (stellar mass: {len(stellar_masses)}, Q: {len(q_values)}, triax: {len(triax_values)})")

    # Print stellar mass to HZ center mapping
    print("Stellar mass → HZ center:")
    for m in stellar_masses:
        _, hz_c, _ = hz_distance_for_mass(m)
        print(f"  {m:.2f} M☉ → {hz_c:.4f} AU")

    results = _execute_parallel(configs, store, args.workers, "Study 10", args.timeout)

    output_file = output_dir / "study10_stellar_mass.json"
    ok_dicts = [r for r in results if r.get("status", "OK") == "OK" or "fractions" in r]
    with open(output_file, "w") as f:
        json.dump(ok_dicts, f, indent=2, default=str)

    print(f"Saved: {output_file}")

    # Summary by stellar mass
    if ok_dicts:
        print("\nPTB fraction by stellar mass:")
        for m_star in stellar_masses:
            mass_results = [r for r in ok_dicts
                           if r["config"].get("stellar_mass_msun", 0.15) == m_star]
            if mass_results:
                mean_ptb = np.mean([r.get("fractions", {}).get("PTB", 0) for r in mass_results])
                print(f"  {m_star:.2f} M☉: {mean_ptb*100:.1f}% PTB (n={len(mass_results)})")

    return ok_dicts


def generate_summary_report(
    output_dir: Path,
    study1,
    study2,
    study3,
    study4,
    study5=None,
    study6=None,
    study7=None,
    study8=None,
    study9=None,
    study10=None,
    study11=None,
):
    """Generate markdown summary report."""
    report_path = output_dir / "rarity_analysis.md"

    rarity_stats = study2[1]
    fit_gaussian = study3[1]
    q_min, q_max = study4[1], study4[2]

    lines = [
        "# Bipolaris v3.4 Sensitivity Analysis",
        "",
        "## Executive Summary",
        "",
        f"**Rarity estimate**: 1:{rarity_stats['rarity']:.0f} systems",
        f"- P(viable) = {rarity_stats['p_viable']:.4f} [{rarity_stats['ci_lower']:.4f}, {rarity_stats['ci_upper']:.4f}]",
        f"- Viable samples: {rarity_stats['n_viable']}/{rarity_stats['n_total']}",
        "",
    ]

    if q_min is not None:
        lines.extend([
            f"**Viable Q range**: [{q_min}, {q_max}]",
            f"- v3.2 baseline (Q=20) is {'WITHIN' if q_min <= 20 <= q_max else 'OUTSIDE'} viable range",
            "",
        ])

    if not np.isnan(fit_gaussian.get("fwhm", float("nan"))):
        lines.extend([
            f"**Resonance window (Gaussian fit)**:",
            f"- FWHM: {fit_gaussian['fwhm']*1000:.2f} mAU",
            f"- Center: {fit_gaussian['center']:.4f} AU",
            f"- R²: {fit_gaussian['r_squared']:.3f}",
            "",
        ])

    lines.extend(["## Credibility Assessment", ""])

    rarity = rarity_stats['rarity']
    if rarity < 100:
        credibility = "HIGH — Common configuration, easily justified"
    elif rarity < 1000:
        credibility = "MODERATE — Acceptable for hard sci-fi (1:100 to 1:1000 range)"
    elif rarity < 10000:
        credibility = "LOW — Requires anthropic selection justification"
    else:
        credibility = "VERY LOW — May need parameter adjustment"

    lines.extend([
        f"**Credibility level**: {credibility}",
        "",
        "### Study 1: Q-Distance Grid",
        f"- Configurations: {len(study1)}",
        f"- Viable configs: {sum(1 for r in study1 if r.get('quality', {}).get('is_viable'))}",
        "",
        "### Study 2: Monte Carlo Rarity",
        f"- Total samples: {rarity_stats['n_total']}",
        f"- Viable samples: {rarity_stats['n_viable']}",
        f"- Rarity: 1:{rarity_stats['rarity']:.0f}",
        "",
        "### Study 3: Resonance Robustness",
    ])

    ok_study3 = [r for r in study3[0] if r.get("fractions")]
    if ok_study3:
        lines.extend([
            f"- Distance range: {min(r['config']['distance_au'] for r in ok_study3):.3f} - "
            f"{max(r['config']['distance_au'] for r in ok_study3):.3f} AU",
            f"- Peak PTB: {max(r['fractions'].get('PTB', 0) for r in ok_study3)*100:.1f}% "
            f"at {fit_gaussian.get('center', 0):.4f} AU",
        ])
    lines.extend([
        "",
        "### Study 4: Q Boundary Detection",
        f"- Q range tested: [5, 100]",
        f"- Viable Q range: [{q_min}, {q_max}]",
        "",
        "## Recommendations",
        "",
    ])

    if rarity > 1000:
        lines.append("- **Consider parameter adjustment**: Rarity exceeds 1:1000 threshold")
    if q_min and 20 < q_min:
        lines.append(f"- **Q=20 too low**: Minimum viable Q is {q_min}")
    if q_max and 20 > q_max:
        lines.append(f"- **Q=20 too high**: Maximum viable Q is {q_max}")
    if fit_gaussian.get("fwhm", 1) < 0.001:
        lines.append("- **Narrow resonance window**: Distance is highly constrained (<1 mAU)")

    if not any(["Consider" in l or "too low" in l or "too high" in l or "Narrow" in l for l in lines[-4:]]):
        lines.append("- **No major concerns**: Current configuration is within viable ranges")

    # Study 5-7 sections (if available)
    if study5:
        study5_results, study5_candidates = study5
        lines.extend([
            "",
            "## Slow Bouncer Analysis",
            "",
            "### Study 5: Slow Bouncer Discovery",
            f"- Configurations tested: {len(study5_results)}",
            f"- Slow bouncer candidates: {len(study5_candidates)}",
        ])
        if study5_candidates:
            top5 = study5_candidates[:5]
            lines.append("- **Top 5 candidates:**")
            for i, c in enumerate(top5, 1):
                lines.append(
                    f"  {i}. Q={c['config']['q']}, dist={c['config']['distance_au']:.4f} AU, "
                    f"score={c['score_details']['slow_bouncer_score']:.3f}"
                )

    if study6:
        lines.extend([
            "",
            "### Study 6: Thermal Optimization",
            f"- Optimized configs: {len(study6)}",
        ])
        if study6:
            best = study6[0]
            lines.extend([
                f"- **Best thermal config:**",
                f"  - Albedo: {best['config']['albedo']}",
                f"  - CO2: {best['config']['co2_pct']}%",
                f"  - T_term: {best['temps'].get('t_term', 'N/A')}K",
            ])

    if study7:
        study7_results, study7_rarity = study7
        lines.extend([
            "",
            "### Study 7: Slow Bouncer Rarity",
            f"- Monte Carlo samples: {study7_rarity['n_total']}",
            f"- Slow bouncers found: {study7_rarity['n_viable']}",
            f"- **Rarity estimate**: 1:{study7_rarity['rarity']:.0f}",
            f"- P(slow_bouncer) = {study7_rarity['p_viable']:.4f} "
            f"[{study7_rarity['ci_lower']:.4f}, {study7_rarity['ci_upper']:.4f}]",
        ])

    if study8:
        study8_results, study8_candidates = study8
        lines.extend([
            "",
            "## Extended Studies",
            "",
            "### Study 8: Low-Q Exploration",
            f"- Q range: 8-14 (below Study 5 minimum)",
            f"- Configurations tested: {len(study8_results)}",
            f"- Slow bouncer candidates: {len(study8_candidates)}",
        ])
        if study8_candidates:
            top3 = study8_candidates[:3]
            lines.append("- **Top candidates:**")
            for i, c in enumerate(top3, 1):
                lines.append(
                    f"  {i}. Q={c['config']['q']}, dist={c['config']['distance_au']:.4f} AU"
                )

    if study9:
        lines.extend([
            "",
            "### Study 9: Fine Distance Resolution",
            f"- Resolution: 100 μAU",
            f"- Configurations tested: {len(study9)}",
        ])

    if study10:
        lines.extend([
            "",
            "### Study 10: Stellar Mass Sweep",
            f"- Stellar mass range: 0.08-0.20 M☉ (M8V to M4V)",
            f"- Configurations tested: {len(study10)}",
        ])
        # Compute mean PTB by stellar mass
        if study10:
            from shared.analysis import hz_distance_for_mass
            for m in [0.08, 0.10, 0.12, 0.15, 0.18, 0.20]:
                mass_results = [r for r in study10
                               if r["config"].get("stellar_mass_msun", 0.15) == m]
                if mass_results:
                    mean_ptb = np.mean([r.get("fractions", {}).get("PTB", 0) for r in mass_results])
                    _, hz_c, _ = hz_distance_for_mass(m)
                    lines.append(f"  - {m:.2f} M☉ (HZ={hz_c:.3f} AU): {mean_ptb*100:.1f}% PTB")

    if study11:
        study11_results, study11_candidates = study11
        lines.extend([
            "",
            "### Study 11: Low-Q Flip-Flop Search",
            f"- Q range: 8-14 at outer distances (0.074-0.076 AU)",
            f"- Configurations tested: {len(study11_results)}",
            f"- **Flip-flop candidates: {len(study11_candidates)}**",
        ])
        if study11_candidates:
            top3 = study11_candidates[:3]
            lines.append("- **Top candidates:**")
            for i, c in enumerate(top3, 1):
                ptb = c['fractions'].get('PTB', 0)
                lines.append(
                    f"  {i}. Q={c['config']['q']}, dist={c['config']['distance_au']:.4f} AU, "
                    f"PTB={ptb*100:.1f}%, score={c['score_details']['slow_bouncer_score']:.3f}"
                )
        else:
            # Show PTB range if no candidates
            ptb_values = [r.get("fractions", {}).get("PTB", 0) for r in study11_results if r.get("fractions")]
            if ptb_values:
                in_range = sum(1 for p in ptb_values if 0.05 <= p <= 0.30)
                lines.append(f"- PTB range: {min(ptb_values)*100:.1f}% - {max(ptb_values)*100:.1f}%")
                lines.append(f"- In target range (5-30%): {in_range}/{len(ptb_values)}")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nSummary report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Bipolaris v3.5 sensitivity analysis")
    parser.add_argument("--output", default="results/sensitivity_analysis_v35/")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--n-orbits", type=int, default=20_000,
                        help="N-body orbits per config (default: 20k for speed)")
    parser.add_argument("--timeout", type=int, default=TIMEOUT_DEFAULT,
                        help=f"Timeout per config in seconds (default: {TIMEOUT_DEFAULT})")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test mode (reduced samples)")
    parser.add_argument("--study", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], default=None,
                        help="Run single study only (default: all)")
    parser.add_argument("--all", action="store_true",
                        help="Run all studies (same as not specifying --study)")
    parser.add_argument("--resume-configs", type=Path,
                        help="Re-run specific configs from file (one config_id per line)")
    parser.add_argument("--study5-candidates", type=Path, default=None,
                        help="Path to Study 5 candidates JSON for Study 6 input")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    print("=== Bipolaris v3.5 Sensitivity Analysis ===")
    print(f"Output directory: {output_dir}")
    print(f"Workers: {args.workers}")
    print(f"N-body orbits: {args.n_orbits:,}")
    print(f"Timeout: {args.timeout}s (risky: {TIMEOUT_RISKY}s)")

    # Handle --resume-configs mode
    if args.resume_configs:
        rerun_specific_configs(args, output_dir)
        return

    study1_results = None
    study2_results = None
    study3_results = None
    study4_results = None
    study5_results = None
    study5_candidates = None
    study6_results = None
    study7_results = None
    study8_results = None
    study8_candidates = None
    study9_results = None
    study10_results = None
    study11_results = None
    study11_candidates = None

    # All studies run by default (or with --all)
    run_default = args.study is None and not args.all

    if args.study == 1 or args.all or run_default:
        study1_results = study1_q_distance_grid(args, output_dir)

    if args.study == 2 or args.all or run_default:
        study2_results = study2_monte_carlo_rarity(args, output_dir)

    if args.study == 3 or args.all or run_default:
        study3_results = study3_resonance_robustness(args, output_dir)

    if args.study == 4 or args.all or run_default:
        study4_results = study4_q_boundary_detection(args, output_dir)

    if args.study == 5 or args.all or run_default:
        study5_results, study5_candidates = study5_slow_bouncer_discovery(args, output_dir)

    if args.study == 6 or args.all or run_default:
        study6_results = study6_thermal_optimization(args, output_dir, study5_candidates)

    if args.study == 7 or args.all or run_default:
        study7_results = study7_rarity_estimation(args, output_dir)

    # Study 8: Low-Q exploration
    if args.study == 8 or args.all or run_default:
        study8_results, study8_candidates = study8_low_q_sweep(args, output_dir)

    # Study 9: Fine distance resolution
    if args.study == 9 or args.all or run_default:
        study9_results = study9_fine_distance_sweep(args, output_dir)

    # Study 10: Stellar mass sweep
    if args.study == 10 or args.all or run_default:
        study10_results = study10_stellar_mass_sweep(args, output_dir)

    # Study 11: Low-Q flip-flop search
    if args.study == 11 or args.all or run_default:
        study11_results, study11_candidates = study11_low_q_flipflop(args, output_dir)

    # Study 12: Bifurcation boundary mapping
    if args.study == 12 or args.all or run_default:
        study12_results = study12_bifurcation_mapping(args, output_dir)

    # Generate report if Studies 1-4 ran (core studies)
    if all([study1_results, study2_results, study3_results, study4_results]):
        generate_summary_report(
            output_dir, study1_results, study2_results, study3_results, study4_results,
            study5=(study5_results, study5_candidates) if study5_results else None,
            study6=study6_results,
            study7=study7_results,
            study8=(study8_results, study8_candidates) if study8_results else None,
            study9=study9_results,
            study10=study10_results,
            study11=(study11_results, study11_candidates) if study11_results else None,
        )

    print(f"\n✓ Analysis complete. Results in: {output_dir}")


if __name__ == "__main__":
    main()
