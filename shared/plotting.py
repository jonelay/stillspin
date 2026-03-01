"""Minimal matplotlib configuration for consistent plot styling."""

import matplotlib.pyplot as plt


def apply_style():
    """Apply shared plot style."""
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "font.family": "serif",
        "savefig.bbox": "tight",
        "savefig.dpi": 200,
    })


def save_and_show(fig, path, show=False):
    """Save figure to path, optionally display."""
    fig.savefig(path)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved: {path}")
