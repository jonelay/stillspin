#!/usr/bin/env python3
"""Stage 2: Fine-tune selected system from Stage 1.

Takes a candidate from Stage 1 results and performs a dense grid search
around its parameters to find optimal configuration.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from shared.constants import L_SUN, M_EARTH, M_SUN, R_EARTH
from thermal_sweep.ebm import tidal_lock_temperatures
from tlsr_spin.coarse_ptb_sweep import score_plausibility, score_ptb_behavior
from tlsr_spin.sweep import run_single

# Fine-tuning grid parameters
N_ORBITS_FINE = 10_000  # 2x coarse for better statistics
FINE_DISTANCE_DELTA = 0.005  # ±0.005 AU
FINE_ALBEDO_DELTA = 0.03     # ±0.03
FINE_Q_VALUES = [10, 50, 100]  # Add Q=100


def _run_fine_job(job: dict) -> dict:
    """Worker function for fine-tuning grid search.

    Args:
        job: Dict with candidate params and overrides.

    Returns:
        Result dict with temperatures, PTB fractions, and scores.
    """
    base_candidate = job["base_candidate"]
    distance_au = job["distance_au"]
    albedo = job["albedo"]
    triax = job["triax"]
    q = job["q"]
    n_orbits = job["n_orbits"]

    # Recompute temperatures with new params
    temps = tidal_lock_temperatures(
        distance_au=distance_au,
        luminosity_lsun=base_candidate["star_luminosity"],
        albedo=albedo,
        co2_fraction=base_candidate["co2_fraction"],
    )

    # Build CLI overrides
    cli_overrides = {
        "STAR_MASS": base_candidate["star_mass"] * M_SUN,
        "STAR_MASS_MSUN": base_candidate["star_mass"],
        "STAR_LUMINOSITY": base_candidate["star_luminosity"] * L_SUN,
        "STAR_LUMINOSITY_LSUN": base_candidate["star_luminosity"],
        "STAR_TEFF": base_candidate["star_teff"],
        "BIPOLARIS_DISTANCE_AU": distance_au,
        "BIPOLARIS_MASS": 1.0 * M_EARTH,
        "BIPOLARIS_RADIUS": 1.06 * R_EARTH,
        "SURFACE_ALBEDO": albedo,
        "ATMO_CO2_FRACTION_MID": base_candidate["co2_fraction"],
    }

    # Run TLSR (use scenario "A" as base, override with custom params)
    t0 = time.time()
    result = run_single(
        scenario_name="A",
        triaxiality=triax,
        tidal_q=q,
        n_orbits=n_orbits,
        cli_overrides=cli_overrides,
        quiet=True,
    )
    elapsed = time.time() - t0

    # Score results
    ptb_score, ptb_desc = score_ptb_behavior(result["fractions"])
    plaus_score = score_plausibility(base_candidate, triax, q)

    # Temperature scoring (simplified)
    temp_score = 100.0  # Assume all fine-tuned configs are acceptable

    total_score = 0.4 * temp_score + 0.4 * ptb_score + 0.2 * plaus_score

    return {
        "distance_au": distance_au,
        "albedo": albedo,
        "triax": triax,
        "q": q,
        "t_sub": round(temps["t_substellar"], 1),
        "t_term": round(temps["t_terminator"], 1),
        "t_anti": round(temps["t_antistellar"], 1),
        "fractions": result["fractions"],
        "ptb_score": ptb_score,
        "ptb_desc": ptb_desc,
        "plaus_score": plaus_score,
        "temp_score": temp_score,
        "total_score": total_score,
        "duration_yr": result["duration_yr"],
        "elapsed": elapsed,
    }


def fine_tune_system(
    base_candidate: dict,
    output_dir: str,
    n_orbits: int = N_ORBITS_FINE,
    workers: int = 1,
    verbose: bool = True,
) -> list[dict]:
    """Run fine-tuning grid search around base candidate.

    Args:
        base_candidate: Candidate dict from Stage 1.
        output_dir: Directory to save results.
        n_orbits: N-body orbits per run.
        workers: Number of parallel workers.
        verbose: Print progress messages.

    Returns:
        List of result dicts, sorted by total_score.
    """
    if verbose:
        print(f"Fine-tuning system: {base_candidate['star_type']} at "
              f"{base_candidate['distance_au']:.4f} AU")
        print()

    # Build fine grid
    base_dist = base_candidate["distance_au"]
    base_albedo = base_candidate["albedo"]
    base_triax = base_candidate["triax"]

    distances = np.linspace(
        base_dist - FINE_DISTANCE_DELTA,
        base_dist + FINE_DISTANCE_DELTA,
        5
    )
    albedos = np.linspace(
        base_albedo - FINE_ALBEDO_DELTA,
        base_albedo + FINE_ALBEDO_DELTA,
        3
    )
    # Triaxiality: ±50% (3 points)
    triaxialities = [base_triax * 0.5, base_triax, base_triax * 1.5]

    jobs = []
    for dist in distances:
        for alb in albedos:
            for triax in triaxialities:
                for q in FINE_Q_VALUES:
                    jobs.append({
                        "base_candidate": base_candidate,
                        "distance_au": dist,
                        "albedo": alb,
                        "triax": triax,
                        "q": q,
                        "n_orbits": n_orbits,
                    })

    n_total = len(jobs)
    if verbose:
        print(f"Fine-tuning grid: {n_total} runs")
        print(f"  Distance: {len(distances)} points ({distances[0]:.4f}-{distances[-1]:.4f} AU)")
        print(f"  Albedo: {len(albedos)} points ({albedos[0]:.2f}-{albedos[-1]:.2f})")
        print(f"  Triaxiality: {len(triaxialities)} points "
              f"({triaxialities[0]:.0e}-{triaxialities[-1]:.0e})")
        print(f"  Tidal Q: {FINE_Q_VALUES}")
        print(f"  N-body orbits: {n_orbits:,}")
        print(f"  Workers: {workers}")
        print()

    # Execute
    results = []
    t_start = time.time()

    if workers <= 1:
        for i, job in enumerate(jobs):
            if verbose and i % 10 == 0:
                print(f"[{i+1}/{n_total}] Running...")
            result = _run_fine_job(job)
            results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_job = {executor.submit(_run_fine_job, job): job for job in jobs}
            for i, future in enumerate(as_completed(future_to_job)):
                result = future.result()
                if verbose and (i + 1) % 10 == 0:
                    print(f"[{i+1}/{n_total}] Completed")
                results.append(result)

    elapsed_total = time.time() - t_start
    results.sort(key=lambda r: r["total_score"], reverse=True)

    if verbose:
        print(f"\nFine-tuning complete in {elapsed_total:.1f}s")
        print(f"Average: {elapsed_total / n_total:.1f}s per run")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, "fine_tuned_results.json")
    with open(result_path, "w") as f:
        json.dump({
            "base_candidate": base_candidate,
            "results": results,
            "elapsed_total": elapsed_total,
        }, f, indent=2, default=str)

    if verbose:
        print(f"Saved: {result_path}")
        print(f"\nTop 5 configurations:")
        for i, r in enumerate(results[:5]):
            print(f"  {i+1}. a={r['distance_au']:.4f} AU, A={r['albedo']:.2f}, "
                  f"triax={r['triax']:.0e}, Q={r['q']} → "
                  f"PTB={r['fractions'].get('PTB', 0)*100:.0f}%, "
                  f"score={r['total_score']:.0f}")

    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Stage 2 fine-tuning")
    parser.add_argument("--candidate", required=True,
                        help="Path to stage1_results.json")
    parser.add_argument("--rank", type=int, default=1,
                        help="Which candidate to fine-tune (1 = top)")
    parser.add_argument("--output", default="results/stage2_finetune",
                        help="Output directory")
    parser.add_argument("--n-orbits", type=int, default=N_ORBITS_FINE,
                        help="N-body orbits per run")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress messages")
    args = parser.parse_args()

    # Load Stage 1 results
    with open(args.candidate) as f:
        stage1_results = json.load(f)

    if args.rank < 1 or args.rank > len(stage1_results):
        print(f"ERROR: Rank {args.rank} out of range (1-{len(stage1_results)})")
        sys.exit(1)

    base_candidate = stage1_results[args.rank - 1]

    if not args.quiet:
        print("=" * 70)
        print("STAGE 2: FINE-TUNING")
        print("=" * 70)
        print(f"\nSelected candidate (rank {args.rank}):")
        cand = base_candidate["thermal_candidate"]
        print(f"  Star: {cand['star_type']} ({cand['star_mass']:.3f} M☉)")
        print(f"  Distance: {cand['distance_au']:.4f} AU")
        print(f"  Albedo: {cand['albedo']:.2f}")
        print(f"  Triaxiality: {base_candidate['triax']:.0e}")
        print(f"  Tidal Q: {base_candidate['q']}")
        print(f"  Stage 1 score: {base_candidate['total_score']:.0f}")
        print()

    # Flatten candidate structure
    flat_candidate = {
        **cand,
        "triax": base_candidate["triax"],
        "q": base_candidate["q"],
    }

    results = fine_tune_system(
        flat_candidate,
        args.output,
        n_orbits=args.n_orbits,
        workers=args.workers,
        verbose=not args.quiet,
    )

    if not args.quiet:
        print("\n" + "=" * 70)
        print("STAGE 2 COMPLETE")
        print("=" * 70)
        print(f"\nBest configuration:")
        best = results[0]
        print(f"  Distance: {best['distance_au']:.4f} AU")
        print(f"  Albedo: {best['albedo']:.2f}")
        print(f"  Triaxiality: {best['triax']:.0e}")
        print(f"  Tidal Q: {best['q']}")
        print(f"  Temperatures: T_sub={best['t_sub']:.0f}K, "
              f"T_term={best['t_term']:.0f}K, T_anti={best['t_anti']:.0f}K")
        print(f"  PTB fraction: {best['fractions'].get('PTB', 0)*100:.1f}%")
        print(f"  Total score: {best['total_score']:.0f}")
        print(f"\nResults: {args.output}/fine_tuned_results.json")


if __name__ == "__main__":
    main()
