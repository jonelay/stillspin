#!/usr/bin/env python3
"""
PTB parameter sweep and phase portrait analysis.

Takes N-body output from a fresh TRAPPIST-1 validation run, identifies PTB
intervals, and either generates phase portraits or re-runs spin integration
with varied parameters (Q, triaxiality, eccentricity scaling) to study how
PTB behavior changes.

Usage:
    # Phase portraits of top PTB episodes
    uv run python tlsr-spin/ptb_sweep.py --fresh --planet 3 --q 10 \
        --n-orbits 100000 --phase-portrait

    # Sweep Q across a single PTB interval
    uv run python tlsr-spin/ptb_sweep.py --fresh --planet 3 --q 10 \
        --n-orbits 100000 --sweep-param q --sweep-values 5,10,20,50,100 --plot

    # Sweep triaxiality
    uv run python tlsr-spin/ptb_sweep.py --fresh --planet 3 --q 10 \
        --n-orbits 100000 --sweep-param triax --sweep-values 1e-6,1e-5,1e-4 --plot
"""

import argparse
import os

import numpy as np

from shared.constants import AU, YEAR
from shared.paths import output_dir_for

from tlsr_spin.plots import plot_phase_portrait, plot_ptb_zoom
from tlsr_spin.regime_classifier import (
    classify_regimes,
    compute_regime_fractions,
    extract_ptb_intervals,
)
from tlsr_spin.spin_integrator import integrate_spin
from tlsr_spin.validate import (
    TRAPPIST1_PLANETS,
    TRAPPIST1_STAR_MASS,
    TRAPPIST1_TRIAXIALITY,
    validate_fresh,
)


def sweep_ptb_interval(
    nbody: dict,
    interval: dict,
    planet: dict,
    sweep_param: str,
    sweep_values: list[float],
    tidal_q: float = 10.0,
    triaxiality: float = 1e-5,
    margin_yr: float = 50.0,
) -> list[dict]:
    """Re-run spin integration over a PTB interval with varied parameters.

    Args:
        nbody: Dict from integrate_and_extract with t, e, n, a arrays.
        interval: Single interval dict from extract_ptb_intervals.
        planet: TRAPPIST1_PLANETS entry.
        sweep_param: Parameter to sweep: "q", "triax", or "ecc_scale".
        sweep_values: Values to sweep over.
        tidal_q: Base tidal Q (used when not sweeping Q).
        triaxiality: Base triaxiality (used when not sweeping triax).
        margin_yr: Context before/after PTB (years).

    Returns:
        List of result dicts per sweep value.
    """
    margin_s = margin_yr * YEAR
    t_lo = interval["t_start"] - margin_s
    t_hi = interval["t_end"] + margin_s

    # Slice N-body arrays to the PTB interval + margin
    mask = (nbody["t"] >= t_lo) & (nbody["t"] <= t_hi)
    if mask.sum() < 100:
        print(f"  Warning: only {mask.sum()} samples in interval, skipping")
        return []

    t_sub = nbody["t"][mask]
    e_sub = nbody["e"][mask]
    n_sub = nbody["n"][mask]
    a_mean = float(np.mean(nbody["a"][mask])) * AU

    results = []
    for val in sweep_values:
        q = tidal_q
        triax = triaxiality
        e_scaled = e_sub

        if sweep_param == "q":
            q = val
        elif sweep_param == "triax":
            triax = val
        elif sweep_param == "ecc_scale":
            e_scaled = e_sub * val
        else:
            raise ValueError(f"Unknown sweep_param: {sweep_param}")

        spin = integrate_spin(
            times=t_sub,
            e_t=e_scaled,
            n_t=n_sub,
            m_star=TRAPPIST1_STAR_MASS,
            m_planet=planet["mass"],
            r_planet=planet["radius"],
            a_mean=a_mean,
            tidal_q=q,
            triaxiality=triax,
        )

        regime_result = classify_regimes(spin["t"], spin["gamma"])
        fractions = compute_regime_fractions(regime_result)

        results.append({
            "sweep_param": sweep_param,
            "sweep_value": val,
            "spin": spin,
            "regime_result": regime_result,
            "fractions": fractions,
        })

        print(f"  {sweep_param}={val}: "
              + ", ".join(f"{k}={v:.1%}" for k, v in fractions.items() if v > 0.001))

    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="PTB zoom-in analysis")
    parser.add_argument("--fresh", action="store_true", required=True,
                        help="Run fresh N-body integration")
    parser.add_argument("--planet", type=int, default=3,
                        help="Planet index, 1-based (default: 3 = planet d)")
    parser.add_argument("--q", type=int, default=10, help="Tidal Q (default: 10)")
    parser.add_argument("--n-orbits", type=int, default=100_000,
                        help="Number of orbits (default: 100000)")
    parser.add_argument("--triax", type=float, default=1e-5,
                        help="Triaxiality (B-A)/C (default: 1e-5)")
    parser.add_argument("--phase-portrait", action="store_true",
                        help="Generate phase portraits of PTB episodes")
    parser.add_argument("--sweep-param", type=str, default=None,
                        choices=["q", "triax", "ecc_scale"],
                        help="Parameter to sweep")
    parser.add_argument("--sweep-values", type=str, default=None,
                        help="Comma-separated sweep values")
    parser.add_argument("--plot", action="store_true",
                        help="Generate plots for sweep results")
    parser.add_argument("--margin", type=float, default=50.0,
                        help="Context margin in years (default: 50)")
    args = parser.parse_args()

    out_dir = output_dir_for(__file__)
    os.makedirs(out_dir, exist_ok=True)

    # Run fresh validation to get N-body + spin data
    print("Running fresh validation...")
    results = validate_fresh(args.planet, args.q, args.n_orbits, args.triax)

    # Extract PTB intervals
    ptb_intervals = extract_ptb_intervals(results["regime_result"])
    print(f"\nFound {len(ptb_intervals)} PTB intervals")

    if not ptb_intervals:
        print("No PTB intervals found. Try different parameters.")
        return

    # Print top intervals
    sorted_ptb = sorted(ptb_intervals, key=lambda x: x["duration_yr"], reverse=True)
    for iv in sorted_ptb[:5]:
        print(f"  #{iv['index']}: {iv['duration_yr']:.2f} yr, "
              f"was {iv['underlying_type']}, "
              f"neighbors: {iv['neighbors']}")

    # Phase portraits
    if args.phase_portrait:
        print("\nGenerating phase portraits...")
        plot_phase_portrait(
            results["spin"]["t"],
            results["spin"]["gamma"],
            results["spin"]["gamma_dot"],
            ptb_intervals,
            os.path.join(out_dir, f"ptb_phase_p{args.planet}_Q{args.q}.png"),
        )
        plot_ptb_zoom(
            results["spin"]["t"],
            results["spin"]["gamma"],
            results["regime_result"],
            ptb_intervals,
            os.path.join(out_dir, f"ptb_zoom_p{args.planet}_Q{args.q}.png"),
            margin_yr=args.margin,
        )

    # Parameter sweep
    if args.sweep_param and args.sweep_values:
        sweep_values = [float(v) for v in args.sweep_values.split(",")]
        planet = TRAPPIST1_PLANETS[args.planet]

        # Use the longest PTB interval
        target = sorted_ptb[0]
        print(f"\nSweeping {args.sweep_param} over PTB #{target['index']} "
              f"({target['duration_yr']:.1f} yr)")

        # Need raw N-body data — re-extract from validate_fresh results
        # validate_fresh returns spin results; we need to re-run N-body
        from tlsr_spin.nbody import build_trappist1_system, integrate_and_extract

        print("  Re-running N-body for sub-interval extraction...")
        sim = build_trappist1_system()
        nbody = integrate_and_extract(sim, args.planet, args.n_orbits)

        sweep_results = sweep_ptb_interval(
            nbody, target, planet, args.sweep_param, sweep_values,
            tidal_q=args.q, triaxiality=args.triax, margin_yr=args.margin,
        )

        if args.plot and sweep_results:
            from tlsr_spin.plots import plot_regime_summary

            summary = {}
            for sr in sweep_results:
                label = f"{sr['sweep_param']}={sr['sweep_value']}"
                summary[label] = {"fractions": sr["fractions"]}

            plot_regime_summary(
                summary,
                os.path.join(out_dir,
                             f"ptb_sweep_{args.sweep_param}_p{args.planet}_Q{args.q}.png"),
                title=f"PTB Sweep: {args.sweep_param} (planet {args.planet}, Q={args.q})",
            )


if __name__ == "__main__":
    main()
