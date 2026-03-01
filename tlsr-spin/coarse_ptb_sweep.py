"""Coarse PTB sweep: validate thermal candidates with TLSR dynamics.

Takes thermal candidates from coarse_thermal_sweep and runs fast TLSR
validation sweeps over triaxiality and tidal Q to identify systems that
show PTB behavior.
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from shared.constants import L_SUN, M_EARTH, M_SUN, R_EARTH
from tlsr_spin.sweep import run_single

# Coarse sweep parameters
# Expanded to capture 10-100 year PTB periods
COARSE_TRIAXIALITIES = [3e-5, 5e-5, 7e-5, 1e-4, 1.5e-4, 2e-4]
COARSE_TIDAL_QS = [5, 10, 20, 50]  # Include Q=5 for faster dynamics
N_ORBITS_COARSE = 20_000  # ~60-100 years at typical HZ distances


def score_ptb_behavior(fractions: dict, duration_yr: float = None) -> tuple[float, str]:
    """Score PTB behavior for a system.

    Narrative requirement: PTB periods of 1-100 years (typical), with shorter
    periods (1-50 yr) preferred for observability.

    Args:
        fractions: Regime fractions from TLSR (e.g., {"TL_ZERO": 0.5, "PTB": 0.2}).
        duration_yr: Integration duration in years (for estimating PTB period).

    Returns:
        Tuple of (score, description).
        Score is 0-100 (higher is better). Ideal is 20-60% PTB with short periods.
    """
    ptb_frac = fractions.get("PTB", 0.0)

    if ptb_frac == 0:
        return 0.0, "no_ptb"

    # Ideal range: 20-60% PTB (balanced dynamics)
    if 0.20 <= ptb_frac <= 0.60:
        score = 100.0
        desc = "ideal"
    elif 0.10 <= ptb_frac < 0.20:
        score = 50.0 + (ptb_frac - 0.10) * 500  # 50-100
        desc = "acceptable_low"
    elif 0.60 < ptb_frac <= 0.80:
        score = 100.0 - (ptb_frac - 0.60) * 250  # 100-50
        desc = "acceptable_high"
    elif ptb_frac < 0.10:
        score = 50.0 * ptb_frac / 0.10  # 0-50
        desc = "marginal_low"
    else:  # > 0.80
        score = max(0, 50.0 - (ptb_frac - 0.80) * 250)  # 50-0
        desc = "marginal_high"

    return score, desc


def score_plausibility(candidate: dict, triax: float, q: int) -> float:
    """Score physical plausibility of system parameters.

    Penalizes extreme values.

    Args:
        candidate: Thermal candidate dict.
        triax: Triaxiality.
        q: Tidal Q.

    Returns:
        Plausibility score 0-100.
    """
    score = 100.0

    # Penalize extreme triaxiality (beyond typical values)
    if triax < 1e-5 or triax > 5e-4:
        score -= 20

    # Penalize very low Q (unrealistic dissipation)
    if q < 10:
        score -= 30

    # Penalize stellar types at edge of M-dwarf sequence
    if candidate["star_mass"] < 0.09:
        score -= 10  # M8V at lower mass limit

    return max(0, score)


def _run_ptb_job(job: dict) -> dict:
    """Worker function for parallel PTB sweep.

    Args:
        job: Dict with thermal_candidate, triax, q keys.

    Returns:
        Result dict with scores and regime fractions.
    """
    candidate = job["thermal_candidate"]
    triax = job["triax"]
    q = job["q"]
    n_orbits = job["n_orbits"]

    # Build CLI overrides for this candidate
    cli_overrides = {
        "STAR_MASS": candidate["star_mass"] * M_SUN,
        "STAR_MASS_MSUN": candidate["star_mass"],
        "STAR_LUMINOSITY": candidate["star_luminosity"] * L_SUN,
        "STAR_LUMINOSITY_LSUN": candidate["star_luminosity"],
        "STAR_TEFF": candidate["star_teff"],
        "BIPOLARIS_DISTANCE_AU": candidate["distance_au"],
        "BIPOLARIS_MASS": 1.0 * M_EARTH,
        "BIPOLARIS_RADIUS": 1.06 * R_EARTH,
        "SURFACE_ALBEDO": candidate["albedo"],
        "ATMO_CO2_FRACTION_MID": candidate["co2_fraction"],
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

    # Score PTB behavior
    ptb_score, ptb_desc = score_ptb_behavior(result["fractions"], result["duration_yr"])
    plaus_score = score_plausibility(candidate, triax, q)

    # Combined score (weighted)
    temp_score = candidate["temp_score"]
    total_score = 0.4 * temp_score + 0.4 * ptb_score + 0.2 * plaus_score

    return {
        "thermal_candidate": candidate,
        "triax": triax,
        "q": q,
        "fractions": result["fractions"],
        "ptb_score": ptb_score,
        "ptb_desc": ptb_desc,
        "plaus_score": plaus_score,
        "temp_score": temp_score,
        "total_score": total_score,
        "duration_yr": result["duration_yr"],
        "elapsed": elapsed,
    }


def run_coarse_ptb_sweep(
    thermal_candidates: list[dict],
    output_path: str,
    n_orbits: int = N_ORBITS_COARSE,
    workers: int = 1,
    verbose: bool = True,
) -> list[dict]:
    """Run coarse PTB sweep on thermal candidates.

    Args:
        thermal_candidates: List of thermal candidate dicts.
        output_path: Path to save results JSON.
        n_orbits: Number of N-body orbits per run.
        workers: Number of parallel workers.
        verbose: Print progress messages.

    Returns:
        List of result dicts, sorted by total_score (descending).
    """
    # Build job list
    jobs = []
    for candidate in thermal_candidates:
        for triax in COARSE_TRIAXIALITIES:
            for q in COARSE_TIDAL_QS:
                jobs.append({
                    "thermal_candidate": candidate,
                    "triax": triax,
                    "q": q,
                    "n_orbits": n_orbits,
                })

    n_total = len(jobs)
    if verbose:
        print(f"\nCoarse PTB sweep: {n_total} TLSR runs")
        print(f"  Thermal candidates: {len(thermal_candidates)}")
        print(f"  Triaxialities: {COARSE_TRIAXIALITIES}")
        print(f"  Tidal Qs: {COARSE_TIDAL_QS}")
        print(f"  N-body orbits per run: {n_orbits:,}")
        print(f"  Workers: {workers}")
        print()

    # Execute sweep
    results = []
    t_start = time.time()

    if workers <= 1:
        for i, job in enumerate(jobs):
            if verbose:
                cand = job["thermal_candidate"]
                print(f"[{i+1}/{n_total}] {cand['star_type']} "
                      f"a={cand['distance_au']:.4f} AU, A={cand['albedo']:.2f}, "
                      f"triax={job['triax']:.0e}, Q={job['q']}")
            result = _run_ptb_job(job)
            if verbose:
                print(f"  PTB={result['fractions'].get('PTB', 0)*100:.0f}%, "
                      f"score={result['total_score']:.0f}, "
                      f"elapsed={result['elapsed']:.1f}s")
            results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_job = {executor.submit(_run_ptb_job, job): job for job in jobs}
            for i, future in enumerate(as_completed(future_to_job)):
                result = future.result()
                cand = result["thermal_candidate"]
                if verbose:
                    print(f"[{i+1}/{n_total}] {cand['star_type']} "
                          f"a={cand['distance_au']:.4f} AU, "
                          f"triax={result['triax']:.0e}, Q={result['q']} → "
                          f"PTB={result['fractions'].get('PTB', 0)*100:.0f}%, "
                          f"score={result['total_score']:.0f}")
                results.append(result)

    elapsed_total = time.time() - t_start

    # Sort by total score (descending)
    results.sort(key=lambda r: r["total_score"], reverse=True)

    if verbose:
        print(f"\nPTB sweep complete in {elapsed_total:.1f}s")
        print(f"Average runtime: {elapsed_total / n_total:.1f}s per run")
        print(f"\nTop 5 systems:")
        for i, r in enumerate(results[:5]):
            cand = r["thermal_candidate"]
            print(f"  {i+1}. {cand['star_type']} at {cand['distance_au']:.4f} AU, "
                  f"A={cand['albedo']:.2f}, triax={r['triax']:.0e}, Q={r['q']} → "
                  f"score={r['total_score']:.0f} "
                  f"(PTB={r['fractions'].get('PTB', 0)*100:.0f}%)")

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    if verbose:
        print(f"\nSaved: {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coarse PTB sweep")
    parser.add_argument("--thermal-candidates",
                        default="results/stage1_sweep/thermal_candidates.json",
                        help="Input thermal candidates JSON")
    parser.add_argument("--output",
                        default="results/stage1_sweep/stage1_results.json",
                        help="Output results JSON")
    parser.add_argument("--n-orbits", type=int, default=N_ORBITS_COARSE,
                        help="N-body orbits per run")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress messages")
    args = parser.parse_args()

    # Load thermal candidates
    with open(args.thermal_candidates) as f:
        thermal_candidates = json.load(f)

    print(f"Loaded {len(thermal_candidates)} thermal candidates from {args.thermal_candidates}")

    run_coarse_ptb_sweep(
        thermal_candidates,
        args.output,
        n_orbits=args.n_orbits,
        workers=args.workers,
        verbose=not args.quiet,
    )
