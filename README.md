# stillspin

Spin dynamics simulations for tidally-locked exoplanets in broken resonance chains.

## System Overview

**Bipolaris** is a modeled Earth-mass planet in the habitable zone of an M5.5V red dwarf, part of a four-planet quasi-resonant chain. For 7+ Gyr, it was stably tidally locked. After a late instability (~500 Mya) ejected a fifth planet, the remaining chain's elevated eccentricities drive **flip-flop behavior** — the substellar point irregularly shifts between two stable orientations on decadal timescales.

| Body | a (AU) | P (days) | Mass | Notes |
|------|--------|----------|------|-------|
| Star | — | — | 0.15 M☉ | M5.5V, 7.5 Gyr |
| b | 0.018 | 2.3 | 0.8 M⊕ | Hot |
| c | 0.032 | 5.4 | 1.4 M⊕ | Interior to HZ |
| **Bipolaris** | 0.0745 | 19.2 | 1.0 M⊕ | Middle HZ, flip-flop zone |
| d | 0.078 | 20.5 | 0.9 M⊕ | In HZ |
| Moon | 5 R⊕ | 18 hr | ~Ceres | Debris moon |
| Companion | 180 AU | — | 0.55 M☉ | K5V, incl 35° |

### Key Physics Findings

- **Flip-flop dynamics** (Q=22, triaxiality=3×10⁻⁵): TL episodes of 10–40 yr typical, occasionally 100+ yr. PTB transitions ~1 yr each, ~39% total time. TL_PI (antistellar lock) dominates at ~47%.
- **PTB is mean-motion-driven**, not eccentricity-driven: perturbations to n(t) from neighboring planets are the primary driver of spin-state transitions.
- **Tidal Q bifurcation**: Sharp PTB onset thresholds exist (e.g., Q~75 for TRAPPIST-1 c). The flip-flop zone spans only ~500 μAU in orbital distance.
- **Triaxiality threshold scales with distance**: Inner planets lock at (B-A)/C ~ 10⁻⁴, outer planets need 5–7×10⁻⁴.

## Demos

| Demo | Library | Question |
|------|---------|----------|
| `tlsr-spin/` | REBOUND + custom | Does the architecture produce TLSR? What are the regime statistics? |
| `thermal-sweep/` | EBM | What surface temperatures and habitability result from CO₂/albedo? |
| `rebound-stability/` | REBOUND | Is the system dynamically stable over Myr timescales? |
| `tidalpy-dissipation/` | TidalPy | How much tidal heating occurs? |
| `helios-atmosphere/` | HELIOS | What CO₂ level matches the temperature regime? |
| `chain-survey/` | REBOUND + tlsr_spin | How common are flip-flop worlds around M-dwarfs? Population synthesis. |

**Archived (deprecated physics):**
- `archive/obliquity-sweep/` — Colombo-Ward obliquity model (wrong for synchronous rotators)
- `archive/vplanet-obliquity/` — VPLanet obliquity evolution (same physics issue)

## Quick Start

```bash
# Install dependencies
uv sync --extra rebound

# Run TLSR quick smoke test (10k orbits, ~2 minutes)
uv run python tlsr-spin/sweep.py --quick

# Run full TLSR sweep (10M orbits × 30 configs, ~hours)
uv run python tlsr-spin/sweep.py

# Run stability check
uv run python rebound-stability/scenario.py --quick
```

## TLSR Pipeline

The `tlsr-spin/` module implements Shakespeare & Steffen (2023) "True Longitudinal Spin Resonance" physics:

### Physics

Tidally locked planets with time-varying eccentricity experience a restoring torque toward synchronous rotation. When secular perturbations from other planets modulate e(t) faster than tidal damping can respond, the planet's spin axis (substellar point) rotates, creating distinct dynamical regimes:

- **TL_ZERO** — Tidally locked at γ=0 (substellar point fixed)
- **TL_PI** — Tidally locked at γ=π (antistellar point to star)
- **SPINNING** — Continuous rotation (prograde or retrograde)
- **PTB** — Perturbed/transitional (<10 yr duration)

### Modules

```
tlsr-spin/
├── physics.py              # Goldreich & Peale (1966) spin ODE
├── spin_integrator.py      # scipy solve_ivp driver
├── regime_classifier.py    # Regime identification
├── period_analysis.py      # Libration/spin period measurement
├── nbody.py               # REBOUND N-body → e(t), n(t)
├── plots.py               # Visualization
├── sweep.py               # Main CLI
└── validate.py            # Paper reproduction (needs SAfinal*.bin)
```

## Chain Survey

The `scripts/run_chain_survey.py` pipeline performs population synthesis of resonant chain systems to answer: **How common are flip-flop worlds around M-dwarfs?**

### Pipeline Stages

1. **Generate** — Create formation-likely resonant chain systems (Kroupa IMF stellar masses, log-normal planet masses, MMR palette, Hill stability filter)
2. **Evolve** — 500 Myr REBOUND integration (Phase A: disk damping, Phase B: free evolution), break detection
3. **Calibrate** — Learn perturbation filter thresholds from first 400 systems
4. **Probe** — 5K-year N-body extraction of orbital perturbations
5. **Spin Survey** — 5K-year spin dynamics wrapping `tlsr_spin` pipeline, classify flip-flop behavior
6. **Report** — Population statistics and flip-flop rate estimation

```bash
# Full pipeline (quick mode)
uv run python scripts/run_chain_survey.py --all --quick --workers 4

# Individual stages
uv run python scripts/run_chain_survey.py --generate 5000 --seed 42
uv run python scripts/run_chain_survey.py --evolve --workers 24
uv run python scripts/run_chain_survey.py --calibrate --workers 24
uv run python scripts/run_chain_survey.py --learn-filter
uv run python scripts/run_chain_survey.py --probe --workers 24
uv run python scripts/run_chain_survey.py --spin-survey --workers 24
uv run python scripts/run_chain_survey.py --report
```

## Sensitivity Analysis

The `scripts/sensitivity_analysis.py` pipeline quantifies parameter sensitivity and rarity:

```bash
# Run all 12 studies with 24 workers
uv run python scripts/sensitivity_analysis.py --all --workers 24

# Run specific study
uv run python scripts/sensitivity_analysis.py --study 5 --workers 24
```

Twelve studies covering core parameter sensitivity, slow bouncer grid search, thermal optimization, Monte Carlo rarity estimation, bifurcation mapping, and M-dwarf stellar mass sweeps.

## Testing

```bash
# Infrastructure tests (~84 tests)
uv run pytest tests/ -v

# TLSR physics tests (~20 tests)
uv run pytest tlsr-spin/tests/ -v

# Chain survey tests (~80 tests)
uv run pytest chain-survey/tests/ -v
```

## Structure

```
shared/
├── constants.py        # System parameters (single source of truth)
├── scenarios.py        # A–J scenario definitions
├── analysis.py         # Viability scoring, slow bouncer analysis
├── sweep_types.py      # SweepConfig/SweepResult/Episode dataclasses
├── result_store.py     # Crash-safe JSONL storage
├── sweep_runner.py     # Subprocess-based timeout
├── paths.py            # Output directory helpers
└── plotting.py         # Matplotlib style

scripts/
├── sensitivity_analysis.py       # 12-study parameter sweep
├── plot_sensitivity_analysis.py  # Sensitivity analysis visualization
├── run_chain_survey.py           # Chain survey orchestrator
└── plot_chain_survey.py          # Chain survey visualization

tests/
└── test_sweep_infrastructure.py  # ~84 tests

tlsr-spin/             # TLSR spin dynamics (imports as tlsr_spin)
thermal-sweep/         # EBM thermal model (imports as thermal_sweep)
chain-survey/          # Resonant chain survey (imports as chain_survey)
rebound-stability/     # Orbital stability (REBOUND + MEGNO)
helios-atmosphere/     # Radiative transfer (HELIOS, CUDA)
tidalpy-dissipation/   # Tidal heating (TidalPy)

archive/
├── obliquity-sweep/   # Deprecated: Colombo-Ward obliquity
└── vplanet-obliquity/ # Deprecated: VPLanet obliquity
```

## License

This project is licensed under GPL-3.0-or-later. See [LICENSE](LICENSE).

### Dependency licenses

| Scope | License | Packages |
|-------|---------|----------|
| Core | BSD-3-Clause | numpy, scipy, matplotlib |
| `rebound` extra | GPL-3.0 | rebound, reboundx |
| `tidalpy` extra | Apache-2.0 | TidalPy |
| `tidalpy` transitive | CC-BY-NC-SA-4.0 | cyrk |
| `vplanet` extra | MIT | vplanet |

The GPL-3.0 license for this project is required by the REBOUND dependency.

## References

- Shakespeare & Steffen 2023, ApJ 959, 170 — "True Longitudinal Spin Resonance and True Polar Wander of Tidally Locked Planets"
- Goldreich & Peale 1966, AJ 71, 425 — "Spin-orbit coupling in the solar system"
- MacDonald & Dawson 2018, AJ 156, 228 — TRAPPIST-1 migration simulations
- Kopparapu et al. 2013, ApJ 765, 131 — Habitable zone boundaries

## Notes

- The Colombo-Ward obliquity model (`archive/obliquity-sweep/`) is deprecated: it assumes J₂ ∝ ω², which vanishes for synchronous rotation. TLSR is the correct physics for tidally locked planets.
- HELIOS requires CUDA and manual installation — see `helios-atmosphere/README.md`
- TLSR validation requires the paper's SAfinal*.bin files (not included) in `tlsr-spin/data/trappist1/`
