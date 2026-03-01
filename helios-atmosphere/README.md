# HELIOS Atmosphere Demo

**Question**: What CO₂ level is consistent with the stated temperature regime?

## What It Does

Runs the HELIOS radiative transfer code to compute temperature-pressure profiles for Bipolaris under three atmospheric scenarios (0.5%, 1%, 2% CO₂). Determines which CO₂ level produces surface temperatures consistent with target temperatures.

## Prerequisites

HELIOS requires CUDA and manual installation:

```bash
# 1. Install CUDA toolkit (>=11.0)
# 2. Clone HELIOS
git clone https://github.com/exoclime/HELIOS.git
cd HELIOS
uv pip install -e .

# 3. Download opacity tables
# See HELIOS documentation for ktable downloads
# Place in helios-atmosphere/input/
```

## Usage

```bash
uv run python helios-atmosphere/scenario.py
```

## Outputs (in `output/`)

- `tp_profiles.png` — Temperature-pressure profiles for each CO₂ scenario
- `surface_temp_vs_co2.png` — Surface temperature as function of CO₂ fraction
- `atmosphere_report.md` — Summary with recommended CO₂ level

## Validation

- T_eq is already ~297 K before greenhouse warming, so the question is how much
  CO₂ fine-tuning is needed to reach the illuminated-pole target (300–320 K)
- At least one CO₂ scenario should bracket the target range

## Input Files

- `input/` — Opacity tables and stellar spectrum (not committed, too large)
  - PHOENIX M3V spectrum or Planck blackbody at 3400 K
  - HELIOS ktable files
