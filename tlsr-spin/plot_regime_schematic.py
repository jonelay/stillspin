#!/usr/bin/env python3
"""
Schematic illustrations of TLSR spin-orbit regimes.

Generates educational figures showing the three spin states
(TL_ZERO, TL_PI, SPINNING) and transitions between them,
using the Goldreich & Peale (1966) pendulum equation.

Usage:
    uv run python tlsr-spin/plot_regime_schematic.py
"""

import numpy as np
from matplotlib.patches import FancyArrowPatch, Circle, Wedge
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from shared.plotting import apply_style, save_and_show
from shared.paths import output_dir_for


# --- Phase portrait of the conservative pendulum ---

def phase_portrait_rhs(
    _t: float,
    y: list[float],
    omega_s_sq: float,
) -> list[float]:
    """Conservative spin-orbit pendulum: γ̈ = -(1/2) ω_s² sin(2γ).

    Args:
        _t: Time (unused, autonomous system).
        y: State [γ, γ̇].
        omega_s_sq: Triaxiality frequency squared.

    Returns:
        [γ̇, γ̈].
    """
    gamma, gamma_dot = y
    gamma_ddot = -0.5 * omega_s_sq * np.sin(2 * gamma)
    return [gamma_dot, gamma_ddot]


def generate_phase_trajectories(
    omega_s: float = 1.0,
    n_libration: int = 8,
    n_circulation: int = 5,
) -> list[dict]:
    """Integrate trajectories for the phase portrait.

    Args:
        omega_s: Triaxiality frequency (sets separatrix width).
        n_libration: Number of libration trajectories per island.
        n_circulation: Number of circulation trajectories.

    Returns:
        List of dicts with 'gamma', 'gamma_dot', and 'type' keys.
    """
    omega_s_sq = omega_s**2
    t_span = (0, 40)
    t_eval = np.linspace(*t_span, 2000)
    trajectories = []

    # Libration around γ=0 (TL_ZERO)
    for amp in np.linspace(0.1, 1.4, n_libration):
        sol = solve_ivp(
            phase_portrait_rhs, t_span, [amp, 0],
            args=(omega_s_sq,), t_eval=t_eval, rtol=1e-10,
        )
        trajectories.append({
            "gamma": sol.y[0], "gamma_dot": sol.y[1], "type": "TL_ZERO",
        })

    # Libration around γ=π (TL_PI)
    for amp in np.linspace(0.1, 1.4, n_libration):
        sol = solve_ivp(
            phase_portrait_rhs, t_span, [np.pi + amp, 0],
            args=(omega_s_sq,), t_eval=t_eval, rtol=1e-10,
        )
        trajectories.append({
            "gamma": sol.y[0], "gamma_dot": sol.y[1], "type": "TL_PI",
        })

    # Circulation (SPINNING) — prograde and retrograde
    for gdot0 in np.linspace(0.6, 1.2, n_circulation):
        for sign in [1, -1]:
            sol = solve_ivp(
                phase_portrait_rhs, t_span, [0, sign * gdot0],
                args=(omega_s_sq,), t_eval=t_eval, rtol=1e-10,
            )
            trajectories.append({
                "gamma": sol.y[0], "gamma_dot": sol.y[1], "type": "SPINNING",
            })

    # Separatrix (approximate — start near unstable equilibrium)
    for eps in [0.01, -0.01]:
        sol = solve_ivp(
            phase_portrait_rhs, t_span, [np.pi / 2 + eps, 0],
            args=(omega_s_sq,), t_eval=t_eval, rtol=1e-12,
        )
        trajectories.append({
            "gamma": sol.y[0], "gamma_dot": sol.y[1], "type": "separatrix",
        })

    return trajectories


def plot_phase_portrait(output_path: str) -> None:
    """Phase portrait of the spin-orbit pendulum.

    Shows libration islands around γ=0 (TL_ZERO) and γ=π (TL_PI),
    with circulation trajectories (SPINNING) outside the separatrix.

    Args:
        output_path: Path to save figure.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {
        "TL_ZERO": "#2196F3",
        "TL_PI": "#FF9800",
        "SPINNING": "#4CAF50",
        "separatrix": "#E91E63",
    }

    trajectories = generate_phase_trajectories()

    for traj in trajectories:
        color = colors[traj["type"]]
        lw = 1.5 if traj["type"] == "separatrix" else 0.8
        ls = "--" if traj["type"] == "separatrix" else "-"
        alpha = 0.9 if traj["type"] == "separatrix" else 0.6
        ax.plot(traj["gamma"], traj["gamma_dot"],
                color=color, linewidth=lw, linestyle=ls, alpha=alpha)

    # Mark equilibria
    ax.plot(0, 0, "o", color="#2196F3", markersize=10, zorder=5,
            label="TL Zero (stable)")
    ax.plot(np.pi, 0, "o", color="#FF9800", markersize=10, zorder=5,
            label="TL Pi (stable)")
    ax.plot(-np.pi, 0, "o", color="#FF9800", markersize=10, zorder=5)
    ax.plot(np.pi / 2, 0, "x", color="#E91E63", markersize=10,
            markeredgewidth=2, zorder=5, label="Unstable (separatrix)")
    ax.plot(-np.pi / 2, 0, "x", color="#E91E63", markersize=10,
            markeredgewidth=2, zorder=5)

    ax.set_xlabel(r"$\gamma$ (rad)")
    ax.set_ylabel(r"$\dot{\gamma}$ (rad/s, normalized)")
    ax.set_title(
        r"Phase portrait: $\ddot{\gamma} = -\frac{1}{2}\omega_s^2 \sin(2\gamma)$"
        "\n"
        "Libration = tidally locked, Circulation = spinning"
    )
    ax.set_xlim(-1.5 * np.pi, 2.5 * np.pi)
    ax.set_ylim(-1.5, 1.5)

    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$",
                         r"$\pi$", r"$3\pi/2$", r"$2\pi$"])

    ax.legend(loc="upper left", fontsize=9)
    save_and_show(fig, output_path)


# --- Synthetic time series with regime transitions ---

def generate_synthetic_tlsr(
    duration_yr: float = 5000,
    dt_yr: float = 0.1,
    omega_s: float = 0.15,
    epsilon: float = 3e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic γ(t) showing TLSR regime transitions.

    Simulates the damped pendulum with a slowly varying eccentricity
    to produce realistic-looking regime transitions.

    Args:
        duration_yr: Total integration time in years.
        dt_yr: Timestep in years.
        omega_s: Triaxiality frequency (normalized).
        epsilon: Tidal dissipation rate (normalized).

    Returns:
        Tuple of (time_yr, gamma) arrays.
    """
    n_steps = int(duration_yr / dt_yr)
    t = np.linspace(0, duration_yr, n_steps)
    gamma = np.zeros(n_steps)
    gamma_dot = np.zeros(n_steps)

    # Start in TL_ZERO
    gamma[0] = 0.1
    gamma_dot[0] = 0.0

    # Slowly varying "effective forcing" to drive transitions
    # Mimics secular eccentricity pumping changing ω_s
    forcing = omega_s * (1 + 0.8 * np.sin(2 * np.pi * t / 1200)
                         + 0.3 * np.sin(2 * np.pi * t / 450))

    for i in range(n_steps - 1):
        ws = forcing[i]
        ddot = -0.5 * ws**2 * np.sin(2 * gamma[i]) - epsilon * gamma_dot[i]
        gamma_dot[i + 1] = gamma_dot[i] + ddot * dt_yr
        gamma[i + 1] = gamma[i] + gamma_dot[i + 1] * dt_yr

    return t, gamma


def classify_synthetic(gamma: np.ndarray) -> np.ndarray:
    """Simple regime classification for synthetic data.

    Args:
        gamma: Spin-orbit angle array.

    Returns:
        Array of regime labels ('TL_ZERO', 'TL_PI', 'SPINNING').
    """
    # Wrap to [-π, π] for classification
    gamma_wrapped = np.mod(gamma + np.pi, 2 * np.pi) - np.pi
    regimes = np.full(len(gamma), "SPINNING", dtype="U10")

    # TL_ZERO: |γ mod 2π| < 1.2
    near_zero = np.abs(gamma_wrapped) < 1.2
    regimes[near_zero] = "TL_ZERO"

    # TL_PI: |γ mod 2π - π| < 1.2
    near_pi = np.abs(np.abs(gamma_wrapped) - np.pi) < 1.2
    regimes[near_pi] = "TL_PI"

    return regimes


def plot_time_series(output_path: str) -> None:
    """Synthetic γ(t) showing regime transitions.

    Illustrates the TL_ZERO → SPINNING → TL_PI → SPINNING → TL_ZERO cycle.

    Args:
        output_path: Path to save figure.
    """
    apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1],
                             sharex=True)

    t, gamma = generate_synthetic_tlsr()
    regimes = classify_synthetic(gamma)

    colors = {
        "TL_ZERO": "#2196F3",
        "TL_PI": "#FF9800",
        "SPINNING": "#4CAF50",
    }

    # Top panel: γ(t)
    ax = axes[0]

    # Draw regime background shading
    for regime_type, color in colors.items():
        mask = regimes == regime_type
        ax.fill_between(t, -30, 30, where=mask, alpha=0.1, color=color,
                        linewidth=0, step="mid")

    ax.plot(t, gamma, color="black", linewidth=0.4, rasterized=True)

    # Reference lines at multiples of π
    for k in range(-8, 9):
        if k == 0:
            ax.axhline(k * np.pi, color="#2196F3", linestyle="--",
                        alpha=0.3, linewidth=0.8)
        elif k % 2 != 0:
            ax.axhline(k * np.pi, color="#FF9800", linestyle="--",
                        alpha=0.3, linewidth=0.8)

    ax.set_ylabel(r"$\gamma$ (rad)")
    ax.set_title(
        "Synthetic TLSR spin history — regime transitions\n"
        r"TL_ZERO ($\gamma \approx 0$): substellar lock  |  "
        r"TL_PI ($\gamma \approx \pi$): antistellar lock  |  "
        "SPINNING: circulating"
    )

    ymin, ymax = gamma.min() - 1, gamma.max() + 1
    ax.set_ylim(ymin, ymax)

    # Custom legend
    from matplotlib.patches import Patch
    patches = [
        Patch(facecolor=colors["TL_ZERO"], alpha=0.3, label="TL Zero (substellar lock)"),
        Patch(facecolor=colors["TL_PI"], alpha=0.3, label="TL Pi (antistellar lock)"),
        Patch(facecolor=colors["SPINNING"], alpha=0.3, label="Spinning (circulating)"),
    ]
    ax.legend(handles=patches, loc="upper left", fontsize=9)

    # Bottom panel: regime timeline
    ax2 = axes[1]
    regime_num = np.where(regimes == "TL_ZERO", 0,
                          np.where(regimes == "TL_PI", 1, 2))
    for regime_type, yval, color in [("TL_ZERO", 0, colors["TL_ZERO"]),
                                      ("TL_PI", 1, colors["TL_PI"]),
                                      ("SPINNING", 2, colors["SPINNING"])]:
        mask = regimes == regime_type
        ax2.fill_between(t, yval - 0.4, yval + 0.4, where=mask,
                         color=color, alpha=0.7, step="mid", linewidth=0)

    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(["TL Zero", "TL Pi", "Spinning"])
    ax2.set_ylim(-0.6, 2.6)
    ax2.set_xlabel("Time (yr, synthetic)")
    ax2.set_title("Regime classification")

    fig.tight_layout()
    save_and_show(fig, output_path)


# --- Top-down planet schematic ---

def _draw_planet_halves(
    ax: plt.Axes,
    cx: float,
    cy: float,
    radius: float,
    gamma: float,
    spinning: bool,
) -> None:
    """Draw a planet with color-coded dayside/nightside hemispheres.

    The star is to the left, so the dayside (substellar) faces left.
    The planet's orientation is rotated by γ relative to the star-planet line.

    Red hemisphere = reference hemisphere "A" (the one with the bulge tip).
    Teal hemisphere = reference hemisphere "B" (opposite side).
    This makes it visually obvious which face points where across regimes.

    Args:
        ax: Matplotlib axes.
        cx: Planet center x.
        cy: Planet center y.
        radius: Planet radius in plot coords.
        gamma: Spin-orbit angle (rad).
        spinning: Whether to show rotation arrow.
    """
    # Dayside = facing star = pointing left = angle 180°
    # The planet's body is rotated by γ from the star-planet line
    # Reference hemisphere A (bulge tip side) starts at angle 0 (toward star)
    # and rotates by γ

    # Hemisphere A: centered on the bulge direction
    # In plot coords, "toward star" = 180°, so bulge points at 180° + γ
    bulge_dir_deg = np.degrees(np.pi + gamma)

    # Hemisphere A (red/warm): half-disk centered on bulge direction
    hemi_a = Wedge(
        (cx, cy), radius, bulge_dir_deg - 90, bulge_dir_deg + 90,
        facecolor="#E57373", alpha=0.45, zorder=2, edgecolor="none",
    )
    # Hemisphere B (teal/cool): the other half
    hemi_b = Wedge(
        (cx, cy), radius, bulge_dir_deg + 90, bulge_dir_deg + 270,
        facecolor="#4DB6AC", alpha=0.45, zorder=2, edgecolor="none",
    )
    ax.add_patch(hemi_a)
    ax.add_patch(hemi_b)

    # Planet outline
    outline = Circle((cx, cy), radius, fill=False, edgecolor="#546E7A",
                      linewidth=1.5, zorder=3)
    ax.add_patch(outline)

    # Hemisphere labels at the center of each half
    label_r = radius * 0.5
    a_angle = np.radians(bulge_dir_deg)
    b_angle = np.radians(bulge_dir_deg + 180)
    ax.text(cx + label_r * np.cos(a_angle), cy + label_r * np.sin(a_angle),
            "A", ha="center", va="center", fontsize=11, fontweight="bold",
            color="#B71C1C", zorder=5)
    ax.text(cx + label_r * np.cos(b_angle), cy + label_r * np.sin(b_angle),
            "B", ha="center", va="center", fontsize=11, fontweight="bold",
            color="#00695C", zorder=5)


def plot_planet_schematic(output_path: str) -> None:
    """Top-down view of planet in each TLSR regime.

    Shows the star, planet with color-coded hemispheres, triaxial bulge
    orientation, and rotation direction for TL_ZERO, TL_PI, and SPINNING.

    Hemisphere A (red) always contains the bulge tip. Hemisphere B (teal)
    is the opposite side. This makes it easy to track which face of the
    planet points at the star across regime transitions.

    Args:
        output_path: Path to save figure.
    """
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    configs = [
        {
            "title": "TL Zero (substellar lock)",
            "subtitle": r"$\gamma \approx 0$: hemisphere A → star",
            "gamma": 0,
            "color": "#2196F3",
            "spinning": False,
        },
        {
            "title": "TL Pi (antistellar lock)",
            "subtitle": r"$\gamma \approx \pi$: hemisphere B → star",
            "gamma": np.pi,
            "color": "#FF9800",
            "spinning": False,
        },
        {
            "title": "Spinning (circulating)",
            "subtitle": r"$\dot{\gamma} \neq 0$: hemispheres drift",
            "gamma": 0.7,
            "color": "#4CAF50",
            "spinning": True,
        },
    ]

    for ax, cfg in zip(axes, configs):
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-2.8, 2.8)
        ax.set_aspect("equal")
        ax.axis("off")

        # Star (to the left)
        star = Circle((-2.2, 0), 0.35, color="#FFC107", zorder=3)
        ax.add_patch(star)
        ax.annotate("Star", (-2.2, -0.7), ha="center", fontsize=8,
                     color="#666666")

        # Planet with color-coded hemispheres
        planet_cx, planet_cy, planet_r = 1.0, 0, 0.8
        _draw_planet_halves(ax, planet_cx, planet_cy, planet_r,
                            cfg["gamma"], cfg["spinning"])

        # Triaxial bulge line
        angle = cfg["gamma"]
        bulge_angle = np.pi + angle
        bulge_len = 1.05
        bulge_dx = bulge_len * np.cos(bulge_angle)
        bulge_dy = bulge_len * np.sin(bulge_angle)

        ax.plot(
            [planet_cx - bulge_dx, planet_cx + bulge_dx],
            [-bulge_dy, bulge_dy],
            color=cfg["color"], linewidth=3.5, solid_capstyle="round",
            zorder=4, alpha=0.7,
        )
        # Bulge tip dot (hemisphere A side)
        ax.plot(planet_cx + bulge_dx, bulge_dy, "o", color=cfg["color"],
                markersize=9, zorder=5)

        # Substellar point marker (always on the star-facing side)
        ax.plot(planet_cx - planet_r + 0.05, 0, "*", color="#FFC107",
                markersize=16, zorder=6)

        # Arrow: star → planet
        ax.annotate(
            "", xy=(planet_cx - planet_r - 0.15, 0), xytext=(-1.75, 0),
            arrowprops=dict(arrowstyle="->", color="#999999", lw=1,
                            linestyle="--"),
        )

        # Spin axis indicator — dot-in-circle (⊙) = axis pointing out of page
        spin_axis = Circle((planet_cx, planet_cy), 0.12, facecolor="white",
                           edgecolor="#333333", linewidth=1.5, zorder=7)
        ax.add_patch(spin_axis)
        ax.plot(planet_cx, planet_cy, ".", color="#333333", markersize=4,
                zorder=8)
        ax.annotate("spin axis\n(out of page)", (planet_cx, planet_cy - 1.15),
                     ha="center", fontsize=7, color="#555555")

        # Dayside/nightside label
        ax.annotate("dayside", (planet_cx - planet_r - 0.05, -0.55),
                     ha="center", fontsize=7, color="#999999")
        ax.annotate("nightside", (planet_cx + planet_r + 0.05, -0.55),
                     ha="center", fontsize=7, color="#999999")

        # Spinning: show rotation arrow
        if cfg["spinning"]:
            arc = FancyArrowPatch(
                (planet_cx + 0.6, 1.1), (planet_cx - 0.6, 1.1),
                connectionstyle="arc3,rad=0.3",
                arrowstyle="->,head_width=5,head_length=5",
                color=cfg["color"], linewidth=2, zorder=6,
            )
            ax.add_patch(arc)
            ax.text(planet_cx, 1.7, "drifting", ha="center", fontsize=10,
                    color=cfg["color"], style="italic")

        ax.set_title(f"{cfg['title']}\n{cfg['subtitle']}",
                     fontsize=11, color=cfg["color"], fontweight="bold")

    # Legend for hemisphere colors
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor="#E57373", alpha=0.5, label="Hemisphere A (bulge tip)"),
        Patch(facecolor="#4DB6AC", alpha=0.5, label="Hemisphere B (opposite)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=2,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "TLSR regimes — view from above north pole "
        r"(spin axis $\odot$ points out of page)" "\n"
        "Planet rotates around spin axis; "
        r"$\gamma$ = angle between triaxial bulge and star-planet line",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    save_and_show(fig, output_path)


def main() -> None:
    """Generate all schematic figures."""
    out_dir = output_dir_for(__file__)

    print("Generating phase portrait...")
    plot_phase_portrait(f"{out_dir}/phase_portrait.png")

    print("Generating synthetic time series...")
    plot_time_series(f"{out_dir}/regime_timeseries_synthetic.png")

    print("Generating planet schematic...")
    plot_planet_schematic(f"{out_dir}/regime_schematic.png")

    print(f"\nAll figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
