# REBOUND Stability Demo

**Question**: Is the Bipolaris system architecture dynamically stable?

## What It Does

Builds a REBOUND N-body simulation of the full system (star, 4 planets, outer companion) and integrates forward using the WHFast integrator. Computes the MEGNO chaos indicator to assess long-term stability.

## Usage

```bash
uv sync --extra rebound
uv run python rebound-stability/scenario.py --quick    # 1,000 year integration
uv run python rebound-stability/scenario.py            # 1 Myr integration
```

## Outputs (in `output/`)

- `orbital_elements.npz` — Time series of semi-major axes, eccentricities
- `megno.png` — MEGNO indicator vs time
- `orbits.png` — Orbital element evolution
- `stability_report.md` — Summary with pass/fail assessment

## Validation

- MEGNO → 2.0 indicates quasi-periodic (stable)
- MEGNO divergence indicates chaos
- No planet ejections (semi-major axis stays bounded)
- Bipolaris at 0.18 AU should be within the HZ for the M3V primary
- Integration should extend to at least 10⁵ yr to cover secular timescales
