#!/usr/bin/env python3
"""Stage 1 orchestrator: thermal sweep → PTB validation → results.

Coordinates the two-stage coarse sweep to identify promising Bipolaris
system configurations.
"""

import argparse
import json
import os
import sys
import time

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thermal_sweep.coarse_thermal_sweep import run_coarse_thermal_sweep
from tlsr_spin.coarse_ptb_sweep import run_coarse_ptb_sweep


def write_stage1_report(
    thermal_candidates: list[dict],
    ptb_results: list[dict],
    output_dir: str,
    elapsed_thermal: float,
    elapsed_ptb: float,
) -> None:
    """Write human-readable summary report for Stage 1.

    Args:
        thermal_candidates: List of thermal candidate dicts.
        ptb_results: List of PTB result dicts (sorted by score).
        output_dir: Directory to write report.
        elapsed_thermal: Thermal sweep elapsed time (seconds).
        elapsed_ptb: PTB sweep elapsed time (seconds).
    """
    report_path = os.path.join(output_dir, "stage1_report.md")

    lines = ["# Stage 1 Coarse Sweep Report\n"]
    lines.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"**Thermal candidates**: {len(thermal_candidates)}")
    lines.append(f"**PTB validation runs**: {len(ptb_results)}")
    lines.append(f"**Elapsed time**: {elapsed_thermal + elapsed_ptb:.1f}s "
                 f"(thermal: {elapsed_thermal:.1f}s, PTB: {elapsed_ptb:.1f}s)\n")

    # Top 10 systems
    lines.append("## Top 10 Systems\n")
    lines.append("| Rank | Star | a (AU) | Albedo | Triax | Q | "
                 "T_sub | T_term | T_anti | PTB% | Score |")
    lines.append("|------|------|--------|--------|-------|---|"
                 "------|--------|--------|------|-------|")

    for i, r in enumerate(ptb_results[:10]):
        cand = r["thermal_candidate"]
        fracs = r["fractions"]
        ptb_pct = fracs.get("PTB", 0) * 100
        lines.append(
            f"| {i+1} | {cand['star_type']} | {cand['distance_au']:.4f} | "
            f"{cand['albedo']:.2f} | {r['triax']:.0e} | {r['q']} | "
            f"{cand['t_sub']:.0f}K | {cand['t_term']:.0f}K | {cand['t_anti']:.0f}K | "
            f"{ptb_pct:.0f}% | {r['total_score']:.0f} |"
        )

    lines.append("\n## Detailed Results (Top 10)\n")

    for i, r in enumerate(ptb_results[:10]):
        cand = r["thermal_candidate"]
        fracs = r["fractions"]

        lines.append(f"### {i+1}. {cand['star_type']} "
                     f"(a={cand['distance_au']:.4f} AU, A={cand['albedo']:.2f})\n")
        lines.append("**Stellar parameters:**")
        lines.append(f"- Mass: {cand['star_mass']:.3f} M☉")
        lines.append(f"- Luminosity: {cand['star_luminosity']:.5f} L☉")
        lines.append(f"- T_eff: {cand['star_teff']} K\n")

        lines.append("**Planetary parameters:**")
        lines.append(f"- Distance: {cand['distance_au']:.4f} AU")
        lines.append(f"- Albedo: {cand['albedo']:.2f}")
        lines.append(f"- CO₂: {cand['co2_fraction']*100:.2f}%\n")

        lines.append("**TLSR dynamics:**")
        lines.append(f"- Triaxiality: {r['triax']:.0e}")
        lines.append(f"- Tidal Q: {r['q']}")
        lines.append(f"- Duration: {r['duration_yr']:.0f} yr\n")

        lines.append("**Temperatures:**")
        lines.append(f"- Substellar: {cand['t_sub']:.0f} K")
        lines.append(f"- Terminator: {cand['t_term']:.0f} K")
        lines.append(f"- Antistellar: {cand['t_anti']:.0f} K\n")

        lines.append("**Regime fractions:**")
        for regime, frac in sorted(fracs.items()):
            lines.append(f"- {regime}: {frac*100:.1f}%")
        lines.append("")

        lines.append("**Scores:**")
        lines.append(f"- Temperature: {r['temp_score']:.0f}/100")
        lines.append(f"- PTB behavior: {r['ptb_score']:.0f}/100 ({r['ptb_desc']})")
        lines.append(f"- Plausibility: {r['plaus_score']:.0f}/100")
        lines.append(f"- **Total**: {r['total_score']:.0f}/100\n")

    # Summary statistics
    lines.append("## Summary Statistics\n")

    star_types = {}
    for r in ptb_results:
        st = r["thermal_candidate"]["star_type"]
        star_types[st] = star_types.get(st, 0) + 1

    lines.append("**Results by stellar type:**")
    for st, count in sorted(star_types.items()):
        lines.append(f"- {st}: {count} configurations tested")

    lines.append("")

    # PTB occurrence
    ptb_systems = sum(1 for r in ptb_results if r["fractions"].get("PTB", 0) > 0)
    lines.append(f"**PTB occurrence**: {ptb_systems}/{len(ptb_results)} "
                 f"({ptb_systems/len(ptb_results)*100:.1f}%) systems show PTB")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nStage 1 report saved: {report_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Stage 1 coarse sweep orchestrator")
    parser.add_argument("--output", default="results/stage1_sweep",
                        help="Output directory for results")
    parser.add_argument("--n-orbits", type=int, default=5_000,
                        help="N-body orbits per PTB run (default: 5k)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers for PTB sweep")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress messages")
    args = parser.parse_args()

    verbose = not args.quiet
    os.makedirs(args.output, exist_ok=True)

    if verbose:
        print("=" * 70)
        print("STAGE 1: COARSE PARAMETER SWEEP")
        print("=" * 70)
        print()

    # Stage 1a: Thermal sweep
    if verbose:
        print("STEP 1: Thermal sweep across stellar types")
        print("-" * 70)

    thermal_path = os.path.join(args.output, "thermal_candidates.json")
    t0_thermal = time.time()
    thermal_candidates = run_coarse_thermal_sweep(thermal_path, verbose=verbose)
    elapsed_thermal = time.time() - t0_thermal

    if len(thermal_candidates) == 0:
        print("\nERROR: No systems passed thermal filters. Adjust targets or sweep parameters.")
        sys.exit(1)

    # Stage 1b: PTB validation
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 2: PTB validation sweep")
        print("-" * 70)

    ptb_path = os.path.join(args.output, "stage1_results.json")
    t0_ptb = time.time()
    ptb_results = run_coarse_ptb_sweep(
        thermal_candidates,
        ptb_path,
        n_orbits=args.n_orbits,
        workers=args.workers,
        verbose=verbose,
    )
    elapsed_ptb = time.time() - t0_ptb

    # Stage 1c: Report generation
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 3: Generate summary report")
        print("-" * 70)

    write_stage1_report(
        thermal_candidates,
        ptb_results,
        args.output,
        elapsed_thermal,
        elapsed_ptb,
    )

    if verbose:
        print("\n" + "=" * 70)
        print("STAGE 1 COMPLETE")
        print("=" * 70)
        print(f"\nTotal elapsed time: {elapsed_thermal + elapsed_ptb:.1f}s")
        print(f"Top system: {ptb_results[0]['thermal_candidate']['star_type']} at "
              f"{ptb_results[0]['thermal_candidate']['distance_au']:.4f} AU "
              f"(score: {ptb_results[0]['total_score']:.0f})")
        print(f"\nResults directory: {args.output}")
        print("  - thermal_candidates.json: Systems passing thermal filters")
        print("  - stage1_results.json: Full PTB validation results")
        print("  - stage1_report.md: Human-readable summary")
        print("\nNext steps:")
        print("  1. Review stage1_report.md to identify promising candidates")
        print("  2. Run Stage 2 fine-tuning on selected systems:")
        print(f"     uv run python scripts/fine_tune_system.py \\")
        print(f"       --candidate {args.output}/stage1_results.json \\")
        print(f"       --rank 1 \\")
        print(f"       --output results/stage2_finetune/")


if __name__ == "__main__":
    main()
