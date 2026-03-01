#!/usr/bin/env python3
"""Visualization for chain survey results.

Usage:
  uv run python scripts/plot_chain_survey.py --input results/chain_survey/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from chain_survey.chain_types import (
    OrbitalResult,
    PerturbationProbe,
    SpinSurveyResult,
    SystemArchitecture,
)


def _load_jsonl(path, cls):
    if not path.exists():
        return []
    results = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                results.append(cls.from_dict(json.loads(line)))
            except Exception:
                continue
    return results


def plot_mmr_distribution(systems, fig_dir):
    """Distribution of MMR types across all generated systems."""
    from chain_survey.chain_generator import MMR_PALETTE

    mmr_counts = {k: 0 for k in MMR_PALETTE}
    for s in systems:
        for mmr in s.config.mmr_sequence:
            mmr_counts[mmr] = mmr_counts.get(mmr, 0) + 1

    labels = list(mmr_counts.keys())
    counts = [mmr_counts[k] for k in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, counts, color="steelblue")
    ax.set_xlabel("MMR Type")
    ax.set_ylabel("Count")
    ax.set_title("MMR Distribution in Generated Systems")

    path = fig_dir / "mmr_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_hz_coverage(systems, fig_dir):
    """HZ planet positions relative to HZ boundaries."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, s in enumerate(systems[:200]):  # Limit for readability
        ax.axhspan(i - 0.4, i + 0.4, xmin=0, xmax=1, alpha=0.05, color="gray")
        # HZ boundaries
        ax.plot([s.hz_inner_au, s.hz_outer_au], [i, i], "g-", alpha=0.3, linewidth=4)
        # Planet positions
        for j, p in enumerate(s.planets):
            color = "red" if j in s.hz_planet_indices else "blue"
            ax.plot(p.distance_au, i, "o", color=color, markersize=3)

    ax.set_xlabel("Distance (AU)")
    ax.set_ylabel("System Index")
    ax.set_title("Planet Positions and HZ Coverage")

    path = fig_dir / "hz_coverage.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_evolution_outcomes(evolutions, fig_dir):
    """Pie chart of evolution outcomes."""
    status_counts = {}
    for r in evolutions:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1

    labels = list(status_counts.keys())
    counts = [status_counts[k] for k in labels]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(counts, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title("Evolution Outcomes")

    path = fig_dir / "evolution_outcomes.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_perturbation_vs_flipflop(probes, spins, fig_dir):
    """Scatter: perturbation strength vs flip-flop classification."""
    spin_lookup = {s.system_id: s for s in spins}

    rms_pass, rms_fail = [], []
    for p in probes:
        spin = spin_lookup.get(p.system_id)
        if spin is None:
            continue
        if spin.is_flipflop:
            rms_pass.append(p.rms_dn_over_n)
        else:
            rms_fail.append(p.rms_dn_over_n)

    fig, ax = plt.subplots(figsize=(8, 5))
    if rms_fail:
        ax.hist(rms_fail, bins=30, alpha=0.6, label="No flip-flop", color="gray")
    if rms_pass:
        ax.hist(rms_pass, bins=30, alpha=0.6, label="Flip-flop", color="red")
    ax.set_xlabel("RMS δn/n")
    ax.set_ylabel("Count")
    ax.set_title("Perturbation Strength vs Flip-Flop")
    ax.legend()

    path = fig_dir / "perturbation_vs_flipflop.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_stellar_mass_vs_flipflop(systems, spins, fig_dir):
    """Flip-flop rate as function of stellar mass."""
    spin_lookup = {s.system_id: s for s in spins}

    mass_bins = np.linspace(0.08, 0.25, 10)
    bin_total = np.zeros(len(mass_bins) - 1)
    bin_ff = np.zeros(len(mass_bins) - 1)

    for s in systems:
        spin = spin_lookup.get(s.system_id)
        if spin is None:
            continue
        idx = np.digitize(s.stellar_mass_msun, mass_bins) - 1
        if 0 <= idx < len(bin_total):
            bin_total[idx] += 1
            if spin.is_flipflop:
                bin_ff[idx] += 1

    rates = np.where(bin_total > 0, bin_ff / bin_total, 0)
    centers = (mass_bins[:-1] + mass_bins[1:]) / 2

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(centers, rates, width=np.diff(mass_bins) * 0.8, color="steelblue")
    ax.set_xlabel("Stellar Mass (M☉)")
    ax.set_ylabel("Flip-Flop Rate")
    ax.set_title("Flip-Flop Rate vs Stellar Mass")

    path = fig_dir / "stellar_mass_vs_flipflop.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Chain survey visualization")
    parser.add_argument("--input", type=str, default="results/chain_survey/",
                        help="Input directory")
    args = parser.parse_args()

    base = Path(args.input)
    fig_dir = base / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    systems = _load_jsonl(base / "systems" / "results.jsonl", SystemArchitecture)
    evolutions = _load_jsonl(base / "evolution" / "results.jsonl", OrbitalResult)
    probes = _load_jsonl(base / "probes" / "results.jsonl", PerturbationProbe)
    spins = _load_jsonl(base / "spin_survey" / "results.jsonl", SpinSurveyResult)

    print(f"Loaded: {len(systems)} systems, {len(evolutions)} evolutions, "
          f"{len(probes)} probes, {len(spins)} spins")

    if systems:
        plot_mmr_distribution(systems, fig_dir)
        plot_hz_coverage(systems, fig_dir)
    if evolutions:
        plot_evolution_outcomes(evolutions, fig_dir)
    if probes and spins:
        plot_perturbation_vs_flipflop(probes, spins, fig_dir)
    if systems and spins:
        plot_stellar_mass_vs_flipflop(systems, spins, fig_dir)

    print(f"\nAll figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()
