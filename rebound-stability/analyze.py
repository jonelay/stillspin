#!/usr/bin/env python3
"""
Post-processing for REBOUND stability output.

Loads saved orbital elements and generates additional analysis plots.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.plotting import apply_style, save_and_show

import matplotlib.pyplot as plt


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    data = np.load(os.path.join(output_dir, "orbital_elements.npz"))

    times = data["times"]
    semi_major = data["semi_major"]
    eccentricities = data["eccentricities"]
    megno = data["megno"]

    apply_style()
    planet_names = ["b", "c", "Bipolaris", "d"]

    # Semi-major axis variation (relative)
    fig, ax = plt.subplots()
    for j, name in enumerate(planet_names):
        a0 = semi_major[0, j]
        ax.plot(times, (semi_major[:, j] - a0) / a0 * 100, label=name)
    ax.set_xlabel("Time (yr)")
    ax.set_ylabel("Δa/a₀ (%)")
    ax.set_title("Relative Semi-Major Axis Variation")
    ax.legend()
    save_and_show(fig, os.path.join(output_dir, "sma_variation.png"))

    # Eccentricity detail for Bipolaris
    fig, ax = plt.subplots()
    ax.plot(times, eccentricities[:, 2])
    ax.set_xlabel("Time (yr)")
    ax.set_ylabel("Eccentricity")
    ax.set_title("Bipolaris Eccentricity Evolution")
    save_and_show(fig, os.path.join(output_dir, "bipolaris_ecc.png"))

    print("Analysis complete.")


if __name__ == "__main__":
    main()
