"""Shared path utilities for output directories."""

import os

SIMS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def output_dir_for(script_file, scenario_name=None):
    """
    Return the output directory for a demo script, creating it if needed.

    Args:
        script_file: __file__ of the calling script (e.g. rebound-stability/scenario.py)
        scenario_name: optional scenario letter (A, B, C) for sweep subdirectory
    """
    base = os.path.join(os.path.dirname(os.path.abspath(script_file)), "output")
    if scenario_name:
        path = os.path.join(base, scenario_name)
    else:
        path = base
    os.makedirs(path, exist_ok=True)
    return path


def results_dir():
    """Return the top-level results/ directory, creating it if needed."""
    path = os.path.join(SIMS_DIR, "results")
    os.makedirs(path, exist_ok=True)
    return path
