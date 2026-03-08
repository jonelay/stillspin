# CLAUDE.md

## Overview

**stillspin** — Astrodynamics simulation demos for the Bipolaris system (Crossover Canyon project). Each subdirectory is an independent demo using a different Python library to model one aspect of the system.

## Setup

```bash
uv sync
# or with all optional backends:
uv sync --extra all
```

HELIOS requires manual installation (CUDA). See `helios-atmosphere/README.md`.

## Structure

- `shared/constants.py` — System parameters (single source of truth)
- `shared/scenarios.py` — Sensitivity scenario definitions (A–H)
- `shared/analysis.py` — Viability scoring, rarity estimation, slow bouncer analysis
- `shared/sweep_types.py` — SweepConfig/SweepResult/Episode dataclasses with validation
- `shared/result_store.py` — Crash-safe JSONL incremental storage
- `shared/sweep_runner.py` — Subprocess-based timeout (kills hung processes)
- `shared/paths.py` — Output directory helpers
- `shared/plotting.py` — Matplotlib style and save helpers
- `scripts/sensitivity_analysis.py` — 12-study parameter sensitivity pipeline
- `scripts/plot_sensitivity_analysis.py` — Visualization for sensitivity results
- `tests/` — Infrastructure tests (sweep robustness, ~84 tests)
- `tlsr-spin/` — TLSR spin dynamics (Shakespeare & Steffen 2023). Imports as `tlsr_spin`.
- `thermal-sweep/` — EBM thermal model. Imports as `thermal_sweep`.
- `rebound-stability/` — Orbital stability (REBOUND + MEGNO)
- `helios-atmosphere/` — Radiative atmosphere model (HELIOS, CUDA)
- `tidalpy-dissipation/` — Tidal dissipation (TidalPy)
- `chain-survey/` — Resonant chain survey (population synthesis). Imports as `chain_survey`.
- `scripts/run_chain_survey.py` — Chain survey CLI orchestrator
- `notes/plans/` — Development plans and specs
- `results/` — Cross-demo summary
- `archive/obliquity-sweep/` — (deprecated) Colombo-Ward obliquity model
- `archive/vplanet-obliquity/` — (deprecated) VPLanet obliquity evolution

## Conventions

- All system parameters come from `shared/constants.py` (single source of truth)
- Each demo has `--quick` for fast smoke tests where applicable
- Do not include co-author lines in commits

## Testing

```bash
# Infrastructure tests (sweep robustness)
uv run pytest tests/ -v

# TLSR physics tests
uv run pytest tlsr-spin/tests/ -v

# Chain survey tests
uv run pytest chain-survey/tests/ -v
```

## Sensitivity Analysis

```bash
# Run all 12 studies with 24 workers
uv run python scripts/sensitivity_analysis.py --all --workers 24

# Run specific study (1-12)
uv run python scripts/sensitivity_analysis.py --study 5 --workers 24

# Supports: --resume (crash recovery), --timeout (per-config limit), --quick (reduced samples)
```

See `results/sensitivity_analysis_v34/` for output.

## Chain Survey

```bash
# Generate systems, evolve, probe, spin survey (quick mode)
uv run python scripts/run_chain_survey.py --all --quick --workers 4

# Individual stages
uv run python scripts/run_chain_survey.py --generate 5000 --seed 42
uv run python scripts/run_chain_survey.py --evolve --workers 24
uv run python scripts/run_chain_survey.py --calibrate --workers 24
uv run python scripts/run_chain_survey.py --learn-filter
uv run python scripts/run_chain_survey.py --probe --workers 24
uv run python scripts/run_chain_survey.py --spin-survey --workers 24
uv run python scripts/run_chain_survey.py --report

# Supports: --resume, --quick, --timeout, --seed
```
