#!/usr/bin/env python3
"""Visualization for Bipolaris v3.2 sensitivity analysis results.

Generates publication-ready figures from the four sensitivity studies.
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.plotting import apply_style


def plot_q_distance_heatmap(results, output_path):
    """Plot Q-Distance parameter space heatmap."""
    apply_style()

    # Extract data
    q_values = sorted(set(r["config"]["q"] for r in results))
    distances = sorted(set(r["config"]["distance_au"] for r in results))

    # Build grids
    ptb_grid = np.zeros((len(distances), len(q_values)))
    score_grid = np.zeros((len(distances), len(q_values)))

    for r in results:
        i = distances.index(r["config"]["distance_au"])
        j = q_values.index(r["config"]["q"])
        ptb_grid[i, j] = r["fractions"].get("PTB", 0)
        score_grid[i, j] = r["quality"]["overall_score"]

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # PTB fraction heatmap
    im1 = axes[0].imshow(ptb_grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                         origin="lower", extent=[q_values[0], q_values[-1],
                                                  distances[0], distances[-1]])
    axes[0].set_xlabel("Tidal Q")
    axes[0].set_ylabel("Distance (AU)")
    axes[0].set_title("PTB Fraction")
    plt.colorbar(im1, ax=axes[0], label="PTB Fraction")

    # Mark baseline
    axes[0].plot(20, 0.070, "k*", markersize=15, label="v3.2 baseline")
    axes[0].legend()

    # Quality score heatmap
    im2 = axes[1].imshow(score_grid, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                         origin="lower", extent=[q_values[0], q_values[-1],
                                                  distances[0], distances[-1]])
    axes[1].set_xlabel("Tidal Q")
    axes[1].set_ylabel("Distance (AU)")
    axes[1].set_title("Overall Quality Score")
    plt.colorbar(im2, ax=axes[1], label="Quality Score")

    # Mark baseline
    axes[1].plot(20, 0.070, "k*", markersize=15, label="v3.2 baseline")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_monte_carlo_distribution(results, rarity_stats, output_path):
    """Plot Monte Carlo score distribution."""
    apply_style()

    scores = [r["quality"]["overall_score"] for r in results]
    ptb_fracs = [r["fractions"].get("PTB", 0) for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Score histogram
    axes[0].hist(scores, bins=50, alpha=0.7, edgecolor="black")
    axes[0].axvline(0.5, color="red", linestyle="--", label="Viability threshold")
    axes[0].set_xlabel("Overall Quality Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Monte Carlo Score Distribution (n={len(scores)})")
    axes[0].legend()

    # PTB fraction vs score scatter
    axes[1].scatter(ptb_fracs, scores, alpha=0.5, s=10)
    axes[1].axhline(0.5, color="red", linestyle="--", label="Viability threshold")
    axes[1].axvline(0.36, color="blue", linestyle="--", label="v3.2 PTB (36%)")
    axes[1].set_xlabel("PTB Fraction")
    axes[1].set_ylabel("Overall Quality Score")
    axes[1].set_title("PTB vs Quality Score")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_resonance_profile(results, fit_gaussian, fit_lorentzian, output_path):
    """Plot resonance robustness profile."""
    apply_style()

    # Extract data
    distances = np.array([r["config"]["distance_au"] for r in results])
    ptb_fracs = np.array([r["fractions"].get("PTB", 0) for r in results])
    scores = np.array([r["quality"]["overall_score"] for r in results])

    # Sort by distance
    idx = np.argsort(distances)
    distances = distances[idx]
    ptb_fracs = ptb_fracs[idx]
    scores = scores[idx]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # PTB fraction
    axes[0].plot(distances, ptb_fracs, "o-", label="PTB fraction")
    axes[0].axvline(0.070, color="red", linestyle="--", label="v3.2 baseline")
    axes[0].axhline(0.36, color="blue", linestyle="--", alpha=0.5, label="v3.2 PTB target")

    # Plot fits if valid (reconstruct functions from parameters)
    if not np.isnan(fit_gaussian["center"]):
        x_fine = np.linspace(distances[0], distances[-1], 200)
        a, x0, sigma = fit_gaussian["params"]
        y_gaussian = a * np.exp(-((x_fine - x0) ** 2) / (2 * sigma ** 2))
        axes[0].plot(x_fine, y_gaussian, "--", alpha=0.7, label=f"Gaussian fit (R²={fit_gaussian['r_squared']:.2f})")

    if not np.isnan(fit_lorentzian["center"]):
        x_fine = np.linspace(distances[0], distances[-1], 200)
        a, x0, gamma = fit_lorentzian["params"]
        y_lorentzian = a * gamma ** 2 / ((x_fine - x0) ** 2 + gamma ** 2)
        axes[0].plot(x_fine, y_lorentzian, ":", alpha=0.7, label=f"Lorentzian fit (R²={fit_lorentzian['r_squared']:.2f})")

    axes[0].set_ylabel("PTB Fraction")
    axes[0].set_title("Resonance Robustness: PTB vs Distance")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Quality score
    axes[1].plot(distances, scores, "o-", color="purple", label="Quality score")
    axes[1].axvline(0.070, color="red", linestyle="--", label="v3.2 baseline")
    axes[1].axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Viability threshold")
    axes[1].set_xlabel("Distance (AU)")
    axes[1].set_ylabel("Overall Quality Score")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_q_boundary(results, q_min, q_max, output_path):
    """Plot Q boundary detection results."""
    apply_style()

    # Group by triaxiality
    triax_values = sorted(set(r["config"]["triax"] for r in results))
    colors = ["blue", "green", "red"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for triax, color in zip(triax_values, colors):
        subset = [r for r in results if r["config"]["triax"] == triax]
        q_vals = [r["config"]["q"] for r in subset]
        ptb_fracs = [r["fractions"].get("PTB", 0) for r in subset]
        scores = [r["quality"]["overall_score"] for r in subset]

        # Sort by Q
        idx = np.argsort(q_vals)
        q_vals = np.array(q_vals)[idx]
        ptb_fracs = np.array(ptb_fracs)[idx]
        scores = np.array(scores)[idx]

        label = f"triax = {triax:.1e}"
        axes[0].plot(q_vals, ptb_fracs, "o-", color=color, label=label)
        axes[1].plot(q_vals, scores, "o-", color=color, label=label)

    # Mark baseline and viable range
    axes[0].axvline(20, color="red", linestyle="--", label="v3.2 baseline (Q=20)")
    if q_min is not None:
        axes[0].axvspan(q_min, q_max, alpha=0.2, color="green", label=f"Viable range [{q_min}, {q_max}]")
    axes[0].set_ylabel("PTB Fraction")
    axes[0].set_title("Q Boundary Detection: PTB vs Tidal Q")
    axes[0].set_xscale("log")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].axvline(20, color="red", linestyle="--", label="v3.2 baseline (Q=20)")
    axes[1].axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Viability threshold")
    if q_min is not None:
        axes[1].axvspan(q_min, q_max, alpha=0.2, color="green", label=f"Viable range [{q_min}, {q_max}]")
    axes[1].set_xlabel("Tidal Q")
    axes[1].set_ylabel("Overall Quality Score")
    axes[1].set_xscale("log")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_study8_low_q_heatmap(results, output_path):
    """Plot Study 8: Low-Q exploration heatmap."""
    apply_style()

    # Extract unique Q and distance values
    q_values = sorted(set(r["config"]["q"] for r in results))
    distances = sorted(set(r["config"]["distance_au"] for r in results))

    # Build PTB grid (average over triax)
    ptb_grid = np.zeros((len(distances), len(q_values)))

    for r in results:
        i = distances.index(r["config"]["distance_au"])
        j = q_values.index(r["config"]["q"])
        ptb_grid[i, j] = r["fractions"].get("PTB", 0)

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(ptb_grid, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1,
                   origin="lower", extent=[q_values[0], q_values[-1],
                                            distances[0], distances[-1]])

    ax.set_xlabel("Tidal Q")
    ax.set_ylabel("Distance (AU)")
    ax.set_title("Study 8: Low-Q Exploration (Q=8-14)")
    plt.colorbar(im, ax=ax, label="PTB Fraction")

    # Mark bifurcation region
    ax.axhline(0.072, color="white", linestyle="--", alpha=0.7, label="Approx. bifurcation")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_study9_fine_distance(results, output_path):
    """Plot Study 9: Fine distance resolution profile."""
    apply_style()

    # Sort by distance
    sorted_results = sorted(results, key=lambda r: r["config"]["distance_au"])

    distances = [r["config"]["distance_au"] for r in sorted_results]
    ptb_fracs = [r["fractions"].get("PTB", 0) for r in sorted_results]
    tl_zero = [r["fractions"].get("TL_ZERO", 0) for r in sorted_results]
    tl_pi = [r["fractions"].get("TL_PI", 0) for r in sorted_results]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(distances, ptb_fracs, "o-", label="PTB", linewidth=1.5, markersize=4)
    ax.plot(distances, tl_zero, "s-", label="TL₀", linewidth=1.5, markersize=4, alpha=0.7)
    ax.plot(distances, tl_pi, "^-", label="TLπ", linewidth=1.5, markersize=4, alpha=0.7)

    ax.set_xlabel("Distance (AU)")
    ax.set_ylabel("Regime Fraction")
    ax.set_title("Study 9: Fine Distance Resolution (100 μAU)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Find and mark bifurcation
    for i in range(1, len(ptb_fracs)):
        if abs(ptb_fracs[i] - ptb_fracs[i-1]) > 0.15:
            ax.axvline(distances[i], color="red", linestyle="--", alpha=0.5,
                      label=f"Bifurcation ~{distances[i]:.4f} AU")
            break

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_study10_stellar_mass(results, output_path):
    """Plot Study 10: Stellar mass sweep scatter."""
    apply_style()

    # Group by stellar mass
    stellar_masses = sorted(set(r["config"].get("stellar_mass_msun", 0.15) for r in results))

    fig, ax = plt.subplots(figsize=(10, 6))

    ptb_by_mass = []
    for m in stellar_masses:
        mass_results = [r for r in results if r["config"].get("stellar_mass_msun", 0.15) == m]
        ptb_vals = [r["fractions"].get("PTB", 0) for r in mass_results]
        ptb_by_mass.append(ptb_vals)

        # Scatter with jitter
        x = [m] * len(ptb_vals)
        ax.scatter(x, ptb_vals, alpha=0.6, s=50)

    # Box plot overlay
    bp = ax.boxplot(ptb_by_mass, positions=stellar_masses, widths=0.01, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_alpha(0.5)

    # Add spectral type labels
    spectral_types = {0.08: "M8V", 0.10: "M6V", 0.12: "M5V", 0.15: "M5.5V", 0.18: "M4V", 0.20: "M3V"}
    for m in stellar_masses:
        if m in spectral_types:
            ax.annotate(spectral_types[m], (m, 0.95), ha="center", fontsize=9, alpha=0.7)

    ax.axvline(0.15, color="red", linestyle="--", alpha=0.7, label="Bipolaris (0.15 M☉)")
    ax.set_xlabel("Stellar Mass (M☉)")
    ax.set_ylabel("PTB Fraction")
    ax.set_title("Study 10: Stellar Mass Sweep (M-dwarf Range)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/sensitivity_analysis_v35/")
    parser.add_argument("--output", default=None,
                        help="Output directory for figures (default: <input>/figures)")
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output or os.path.join(input_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)

    print("=== Plotting Sensitivity Analysis Results ===")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")

    # Load results
    try:
        with open(os.path.join(input_dir, "study1_q_distance_grid.json")) as f:
            study1 = json.load(f)
        print(f"Study 1: {len(study1)} configs")
        plot_q_distance_heatmap(study1, os.path.join(output_dir, "study1_q_distance_heatmap.png"))
    except FileNotFoundError:
        print("Study 1 results not found, skipping...")

    try:
        with open(os.path.join(input_dir, "study2_monte_carlo.json")) as f:
            study2 = json.load(f)
        print(f"Study 2: {len(study2['results'])} configs")
        plot_monte_carlo_distribution(study2["results"], study2["rarity_stats"],
                                       os.path.join(output_dir, "study2_monte_carlo.png"))
    except FileNotFoundError:
        print("Study 2 results not found, skipping...")

    try:
        with open(os.path.join(input_dir, "study3_resonance_profile.json")) as f:
            study3 = json.load(f)
        print(f"Study 3: {len(study3['results'])} configs")
        plot_resonance_profile(study3["results"], study3["fit_gaussian"], study3["fit_lorentzian"],
                               os.path.join(output_dir, "study3_resonance_profile.png"))
    except FileNotFoundError:
        print("Study 3 results not found, skipping...")

    try:
        with open(os.path.join(input_dir, "study4_q_boundary.json")) as f:
            study4 = json.load(f)
        print(f"Study 4: {len(study4['results'])} configs")
        plot_q_boundary(study4["results"], study4["q_min"], study4["q_max"],
                        os.path.join(output_dir, "study4_q_boundary.png"))
    except FileNotFoundError:
        print("Study 4 results not found, skipping...")

    # Extended studies (8-10)
    try:
        with open(os.path.join(input_dir, "study8_low_q_sweep.json")) as f:
            study8 = json.load(f)
        print(f"Study 8: {len(study8)} configs")
        plot_study8_low_q_heatmap(study8, os.path.join(output_dir, "study8_low_q_heatmap.png"))
    except FileNotFoundError:
        print("Study 8 results not found, skipping...")

    try:
        with open(os.path.join(input_dir, "study9_fine_distance.json")) as f:
            study9 = json.load(f)
        print(f"Study 9: {len(study9)} configs")
        plot_study9_fine_distance(study9, os.path.join(output_dir, "study9_fine_distance.png"))
    except FileNotFoundError:
        print("Study 9 results not found, skipping...")

    try:
        with open(os.path.join(input_dir, "study10_stellar_mass.json")) as f:
            study10 = json.load(f)
        print(f"Study 10: {len(study10)} configs")
        plot_study10_stellar_mass(study10, os.path.join(output_dir, "study10_stellar_mass.png"))
    except FileNotFoundError:
        print("Study 10 results not found, skipping...")

    print(f"\n✓ Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
