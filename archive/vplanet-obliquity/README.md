# VPLanet Obliquity Demo

> **DEPRECATED (2026-02-05)**: This demo uses the Colombo-Ward obliquity model, which
> assumes J₂ ∝ ω². For synchronously rotating planets (ω ≈ n), J₂ vanishes, making this
> physics invalid. The current implementation uses **TLSR (True Longitudinal Spin Resonance)**
> in `tlsr-spin/`, which models spin-orbit desynchronization rather than obliquity oscillations.
> See Shakespeare & Steffen 2023 for the correct physics.

**Question**: What obliquity oscillation period does the moon/perturber configuration produce?

## What It Does

Generates VPLanet `.in` configuration files from shared constants and runs an obliquity evolution simulation using VPLanet's DistRot and EqTide modules. Measures the resulting obliquity oscillation period.

## Usage

```bash
pip install -e ".[vplanet]"
python run.py --quick    # 100 kyr integration
python run.py            # 10 Myr integration
```

## Outputs (in `output/`)

- `obliquity_evolution.png` — Obliquity vs time
- `spin_rate.png` — Rotation rate evolution
- `period_measurement.md` — Measured oscillation period

## Validation

- Obliquity should oscillate between bounds
- Target oscillation period: 3–100 yr
- An outer gas giant as a secular perturber; VPLanet's DistOrb
  module should handle it natively

## Limitations

If VPLanet's built-in secular perturbation theory cannot reproduce the expected
forcing from the outer perturber at 3 AU, an analytic torque workaround is documented in
the output.
