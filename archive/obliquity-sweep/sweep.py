#!/usr/bin/env python3
"""
4D parameter sweep for obliquity oscillations.

Sweeps mutual inclination, perturber distance, moon mass, and moon
orbital distance to find configurations producing obliquity oscillations
matching target requirements.

Usage:
    python sweep.py [--quick] [--config path/to/config.yaml]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from physics import integrate_obliquity, measure_oscillation
from problem import evaluate_grid


def load_config(config_path, quick=False):
    """Load YAML config, applying --quick overrides if requested."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if quick and 'quick' in config:
        quick_overrides = config['quick']
        _deep_merge(config, quick_overrides)

    return config


def _deep_merge(base, override):
    """Recursively merge override into base dict (mutates base)."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def write_results_json(results, config, output_dir):
    """Write results to JSON."""
    def to_list(arr):
        """Convert ndarray to nested list, replacing NaN with None."""
        if arr.ndim == 0:
            return None if np.isnan(arr) else float(arr)
        flat = arr.flatten()
        out = [None if np.isnan(x) else float(x) for x in flat]
        # Reshape back to nested lists
        shaped = np.array(out, dtype=object).reshape(arr.shape)
        return shaped.tolist()

    data = {
        'config': {
            'system': config['system'],
            'integration': {k: v for k, v in config['integration'].items()},
            'targets': config['targets'],
        },
        'grid': {
            'incl_deg': results['incl_vals'].tolist(),
            'kerb_au': results['kerb_vals'].tolist(),
            'masses_mearth': results['masses'].tolist(),
            'distances_rp': results['distances'].tolist(),
        },
        'results': {
            'period_yr': to_list(results['period']),
            'amplitude_deg': to_list(results['amplitude']),
            'alpha_over_s': to_list(results['alpha_over_s']),
            'i_forced_deg': to_list(results['i_forced_deg']),
        },
    }

    path = os.path.join(output_dir, 'results.json')
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Wrote {path}")


def generate_heatmaps(results, config, output_dir):
    """Generate 2D heatmap panels for each (I_mutual, a_kerb) slice."""
    incl_vals = results['incl_vals']
    kerb_vals = results['kerb_vals']
    masses = results['masses']
    dists = results['distances']
    period = results['period']
    amplitude = results['amplitude']
    alpha_over_s = results['alpha_over_s']
    i_forced_deg = results['i_forced_deg']
    targets = config['targets']

    ni, nk = len(incl_vals), len(kerb_vals)

    for ii, incl in enumerate(incl_vals):
        for ik, kerb in enumerate(kerb_vals):
            amp_slice = amplitude[ii, ik, :, :]
            per_slice = period[ii, ik, :, :]
            as_slice = alpha_over_s[ii, ik, :, :]
            if_slice = i_forced_deg[ii, ik, :, :]

            viable = (
                (per_slice >= targets['period_yr']['min']) &
                (per_slice <= targets['period_yr']['max']) &
                (amp_slice >= targets['amplitude_deg']['min']) &
                (amp_slice <= targets['amplitude_deg']['max'])
            )

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'I_mutual={incl:.0f}°  a_kerb={kerb:.1f} AU', fontsize=14)

            # (a) Amplitude
            ax = axes[0, 0]
            vmax = max(np.nanmax(amp_slice), 1) if np.any(np.isfinite(amp_slice)) else 1
            im = ax.pcolormesh(dists, masses, amp_slice,
                               cmap='magma', shading='nearest', vmin=0, vmax=vmax)
            fig.colorbar(im, ax=ax, label='Amplitude (°)')
            ax.set_yscale('log')
            ax.set_ylabel('Moon mass (M⊕)')
            ax.set_title('(a) Oscillation amplitude')

            # (b) Period
            ax = axes[0, 1]
            per_plot = np.where(np.isfinite(per_slice), per_slice, np.nan)
            vmin_p = max(np.nanmin(per_plot), 0.1) if np.any(np.isfinite(per_plot)) else 1
            vmax_p = np.nanmax(per_plot) if np.any(np.isfinite(per_plot)) else 1e5
            im = ax.pcolormesh(dists, masses, per_plot,
                               norm=LogNorm(vmin=vmin_p, vmax=vmax_p),
                               cmap='viridis', shading='nearest')
            fig.colorbar(im, ax=ax, label='Period (yr)')
            ax.set_yscale('log')
            ax.set_title('(b) Oscillation period')

            # (c) α/s ratio
            ax = axes[1, 0]
            a_s = np.where(np.isfinite(as_slice), as_slice, np.nan)
            vmin_as = max(np.nanmin(a_s), 0.01) if np.any(np.isfinite(a_s)) else 0.01
            vmax_as = np.nanmax(a_s) if np.any(np.isfinite(a_s)) else 100
            im = ax.pcolormesh(dists, masses, a_s,
                               norm=LogNorm(vmin=vmin_as, vmax=vmax_as),
                               cmap='coolwarm', shading='nearest')
            fig.colorbar(im, ax=ax, label='α/s')
            try:
                ax.contour(dists, masses, a_s, levels=[1.0], colors='black', linewidths=2)
            except Exception:
                pass
            ax.set_yscale('log')
            ax.set_ylabel('Moon mass (M⊕)')
            ax.set_xlabel('Moon distance (Rp)')
            ax.set_title('(c) α/s ratio')

            # (d) Viability
            ax = axes[1, 1]
            im = ax.pcolormesh(dists, masses, viable.astype(float),
                               cmap='RdYlGn', shading='nearest', vmin=0, vmax=1)
            fig.colorbar(im, ax=ax, label='Viable')
            # Mark model-validity boundary
            if_mean = np.nanmean(if_slice)
            if if_mean > 10:
                ax.text(0.5, 0.02, f'⚠ I_forced≈{if_mean:.1f}° (secular approx. marginal)',
                        transform=ax.transAxes, ha='center', fontsize=8,
                        color='orange', weight='bold')
            ax.set_yscale('log')
            ax.set_xlabel('Moon distance (Rp)')
            ax.set_title(f'(d) Viable ({np.sum(viable)}/{viable.size})')

            plt.tight_layout()
            fname = f'heatmap_I{incl:.0f}_K{kerb:.1f}.png'
            path = os.path.join(output_dir, fname)
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"Wrote {path}")


def generate_summary_table(results, config):
    """Return lines for the summary viability table across all (I_mutual, a_kerb) slices."""
    incl_vals = results['incl_vals']
    kerb_vals = results['kerb_vals']
    period = results['period']
    amplitude = results['amplitude']
    i_forced_deg = results['i_forced_deg']
    targets = config['targets']

    lines = []
    lines.append('| I_mutual (°) | a_kerb (AU) | Viable | Max Amp (°) | I_forced (°) | Model Validity |')
    lines.append('|--------------|-------------|--------|-------------|--------------|----------------|')

    for ii, incl in enumerate(incl_vals):
        for ik, kerb in enumerate(kerb_vals):
            per_slice = period[ii, ik, :, :]
            amp_slice = amplitude[ii, ik, :, :]
            if_slice = i_forced_deg[ii, ik, :, :]

            viable = (
                (per_slice >= targets['period_yr']['min']) &
                (per_slice <= targets['period_yr']['max']) &
                (amp_slice >= targets['amplitude_deg']['min']) &
                (amp_slice <= targets['amplitude_deg']['max'])
            )
            n_viable = int(np.sum(viable))
            max_amp = np.nanmax(amp_slice) if np.any(np.isfinite(amp_slice)) else 0
            mean_if = np.nanmean(if_slice) if np.any(np.isfinite(if_slice)) else 0

            if mean_if > 10:
                validity = 'Extrapolated'
            elif mean_if > 5:
                validity = 'Marginal'
            else:
                validity = 'Reliable'

            lines.append(f'| {incl:.0f} | {kerb:.1f} | {n_viable} | {max_amp:.2f} | {mean_if:.1f} | {validity} |')

    return lines


def generate_report(results, config, output_dir):
    """Generate comprehensive sweep_report.md."""
    incl_vals = results['incl_vals']
    kerb_vals = results['kerb_vals']
    masses = results['masses']
    dists = results['distances']
    period = results['period']
    amplitude = results['amplitude']
    alpha_over_s = results['alpha_over_s']
    i_forced_deg = results['i_forced_deg']
    targets = config['targets']
    system = config['system']
    integ = config['integration']
    sw = config['sweep']

    ni, nk, nm, nd = len(incl_vals), len(kerb_vals), len(masses), len(dists)
    total_cells = ni * nk * nm * nd

    # Global viability
    viable = (
        (period >= targets['period_yr']['min']) &
        (period <= targets['period_yr']['max']) &
        (amplitude >= targets['amplitude_deg']['min']) &
        (amplitude <= targets['amplitude_deg']['max'])
    )
    n_viable = int(np.sum(viable))

    L = []
    def w(s=''):
        L.append(s)

    w('# 4D Obliquity Parameter Sweep Report')
    w()
    w('Colombo-Ward secular spin-orbit model. Sweeps mutual inclination,')
    w('perturber distance, moon mass, and orbital distance to find obliquity')
    w('oscillation regimes matching target temperature requirements.')
    w()

    # Configuration
    w('## Configuration')
    w()
    w('### System Parameters')
    w()
    w('| Parameter | Value |')
    w('|-----------|-------|')
    for k, v in sorted(system.items()):
        w(f'| {k} | {v} |')
    w()

    w('### Sweep Grid')
    w()
    w(f'- Mutual inclination: {sw["mutual_inclination_deg"]["min"]}–{sw["mutual_inclination_deg"]["max"]}° '
      f'({sw["mutual_inclination_deg"]["points"]} points)')
    w(f'- perturber distance: {sw["kerberos_distance_au"]["min"]}–{sw["kerberos_distance_au"]["max"]} AU '
      f'({sw["kerberos_distance_au"]["points"]} points)')
    w(f'- Moon mass: {sw["moon_mass"]["min"]}–{sw["moon_mass"]["max"]} M_earth '
      f'({sw["moon_mass"]["points"]} points, {sw["moon_mass"].get("scale", "linear")})')
    w(f'- Moon distance: {sw["moon_distance"]["min"]}–{sw["moon_distance"]["max"]} Rp '
      f'({sw["moon_distance"]["points"]} points)')
    w(f'- Total cells: {total_cells}')
    w()

    w('### Targets')
    w()
    w(f'- Period: {targets["period_yr"]["min"]}–{targets["period_yr"]["max"]} yr')
    w(f'- Amplitude: {targets["amplitude_deg"]["min"]}–{targets["amplitude_deg"]["max"]}°')
    w()

    # Summary table
    w('## Viability by (I_mutual, a_kerb) Slice')
    w()
    for line in generate_summary_table(results, config):
        w(line)
    w()

    # Heatmaps
    w('## Heatmaps')
    w()
    for ii, incl in enumerate(incl_vals):
        for ik, kerb in enumerate(kerb_vals):
            fname = f'heatmap_I{incl:.0f}_K{kerb:.1f}.png'
            w(f'### I_mutual={incl:.0f}°, a_kerb={kerb:.1f} AU')
            w()
            w(f'![{fname}]({fname})')
            w()

    # Viable configurations detail
    w('## Viable Configurations')
    w()
    w(f'**{n_viable}/{total_cells}** cells fall within the target window.')
    w()

    if n_viable > 0:
        w('| I_mut (°) | a_kerb (AU) | Mass (M⊕) | Dist (Rp) | Period (yr) | Amp (°) | α/s | I_forced (°) |')
        w('|-----------|-------------|-----------|-----------|-------------|---------|-----|--------------|')
        viable_idx = np.argwhere(viable)
        # Show up to 30
        for idx in viable_idx[:30]:
            ii, ik, im, id_ = idx
            w(f'| {incl_vals[ii]:.0f} | {kerb_vals[ik]:.1f} | {masses[im]:.4f} | {dists[id_]:.1f} | '
              f'{period[tuple(idx)]:.1f} | {amplitude[tuple(idx)]:.1f} | '
              f'{alpha_over_s[tuple(idx)]:.3f} | {i_forced_deg[tuple(idx)]:.1f} |')
        if len(viable_idx) > 30:
            w(f'... and {len(viable_idx) - 30} more')
    else:
        w('**No cells fell within the target window.**')
        w()
        w('Top 10 by amplitude:')
        w()
        w('| I_mut (°) | a_kerb (AU) | Mass (M⊕) | Dist (Rp) | Period (yr) | Amp (°) | α/s | I_forced (°) |')
        w('|-----------|-------------|-----------|-----------|-------------|---------|-----|--------------|')
        flat_amp = amplitude.flatten()
        finite_mask = np.isfinite(flat_amp)
        if np.any(finite_mask):
            finite_indices = np.where(finite_mask)[0]
            sorted_idx = np.argsort(flat_amp[finite_mask])[::-1][:10]
            for k in sorted_idx:
                flat_k = finite_indices[k]
                idx = np.unravel_index(flat_k, amplitude.shape)
                ii, ik, im, id_ = idx
                w(f'| {incl_vals[ii]:.0f} | {kerb_vals[ik]:.1f} | {masses[im]:.4f} | {dists[id_]:.1f} | '
                  f'{period[idx]:.1f} | {amplitude[idx]:.2f} | '
                  f'{alpha_over_s[idx]:.3f} | {i_forced_deg[idx]:.1f} |')
    w()

    # Grid-wide stats
    w('## Grid-Wide Statistics')
    w()
    w('| Metric | Min | Max | Median |')
    w('|--------|-----|-----|--------|')
    for name, arr in [('Period (yr)', period), ('Amplitude (°)', amplitude),
                      ('α/s', alpha_over_s), ('I_forced (°)', i_forced_deg)]:
        if np.any(np.isfinite(arr)):
            w(f'| {name} | {np.nanmin(arr):.3g} | {np.nanmax(arr):.3g} | {np.nanmedian(arr):.3g} |')
        else:
            w(f'| {name} | N/A | N/A | N/A |')
    w()

    # Model validity
    w('## Model Validity Notes')
    w()
    w('The forced inclination I_forced is computed from lowest-order Laplace-Lagrange')
    w('secular theory: I_forced = (M_kerb/M_star) × (a_orb/a_kerb)² × sin(I_mutual).')
    w()
    w('- **I_forced < 5°**: Secular approximation reliable.')
    w('- **5° < I_forced < 10°**: Results should be treated as approximate.')
    w('- **I_forced > 10°**: Higher-order terms significant; results are qualitative')
    w('  indicators of which parameter region works, not precise predictions.')
    w()

    # Caveats
    w('## Caveats')
    w()
    w('- Secular theory only (no MMR, no chaotic layers)')
    w('- No tidal orbital evolution of moon')
    w('- Static J₂ (no ice-loading feedback)')
    w('- Forced inclination approximate at high I_mutual')
    w('')

    path = os.path.join(output_dir, 'sweep_report.md')
    with open(path, 'w') as f:
        f.write('\n'.join(L))
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description='4D parameter sweep for obliquity oscillations')
    parser.add_argument('--quick', action='store_true', help='Use reduced grid and shorter integration')
    parser.add_argument('--config', default=None, help='Path to config YAML')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    config_path = args.config or (script_dir / 'config.yaml')
    if not Path(config_path).exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading config: {config_path}")
    config = load_config(config_path, quick=args.quick)

    sw = config['sweep']
    n_incl = sw.get('mutual_inclination_deg', {}).get('points', 1)
    n_kerb = sw.get('kerberos_distance_au', {}).get('points', 1)
    n_mass = sw['moon_mass']['points']
    n_dist = sw['moon_distance']['points']
    total = n_incl * n_kerb * n_mass * n_dist
    print(f"Grid: {n_incl} × {n_kerb} × {n_mass} × {n_dist} = {total} evaluations")
    print(f"Integration: {config['integration']['t_stop_yr']} yr\n")

    print("Running sweep...")
    results = evaluate_grid(config)

    output_dir = script_dir / 'output'
    output_dir.mkdir(exist_ok=True)
    output_dir = str(output_dir)

    print("\nGenerating outputs...")
    write_results_json(results, config, output_dir)
    generate_heatmaps(results, config, output_dir)
    generate_report(results, config, output_dir)

    # Summary
    period = results['period']
    amplitude = results['amplitude']
    targets = config['targets']
    viable = (
        (period >= targets['period_yr']['min']) &
        (period <= targets['period_yr']['max']) &
        (amplitude >= targets['amplitude_deg']['min']) &
        (amplitude <= targets['amplitude_deg']['max'])
    )
    n_viable = int(np.sum(viable))
    n_total = period.size
    print(f"\n{'='*50}")
    print(f"Viable: {n_viable}/{n_total} grid points")
    if np.any(np.isfinite(period)):
        print(f"Period range: {np.nanmin(period):.1f}–{np.nanmax(period):.1f} yr")
    if np.any(np.isfinite(amplitude)):
        print(f"Amplitude range: {np.nanmin(amplitude):.2f}–{np.nanmax(amplitude):.2f}°")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
