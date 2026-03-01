# TidalPy Dissipation Demo

**Question**: How significant are the moon's tides on Bipolaris?

> **Note**: The moon is included for tidal heating calculations but no longer drives the spin
> dynamics. The current TLSR model uses neighboring planet perturbations instead of
> Colombo-Ward obliquity oscillations.

## What It Does

Models Bipolaris as a layered terrestrial body (iron core + silicate mantle) and computes tidal response to lunar gravitational forcing using TidalPy's Andrade/Maxwell rheology.

## Usage

```bash
uv sync --extra tidalpy
uv run python tidalpy-dissipation/scenario.py
```

## Outputs (in `output/`)

- `love_numbers.md` — Computed k₂, h₂ Love numbers
- `tidal_comparison.png` — Bipolaris-Moon vs Earth-Moon tidal amplitude
- `heating_rate.md` — Volumetric and surface tidal heating rates

## Validation

- Love number k₂ ≈ 0.3 (Earth-like interior)
- Moon is 0.015 M⊕ at 20 R⊕ — tidal amplitude is meter-scale
- Meter-scale equilibrium tides are above the linear-response regime; results
  should be interpreted as order-of-magnitude
- Tidal inspiral timescale should be computed to verify Moon's orbit is
  stable over geological time
