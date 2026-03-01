# TLSR Spin Dynamics Model

## Overview

Implements the Goldreich & Peale (1966) spin-orbit equation as used in
the Shakespeare & Steffen (2023) DayNite pipeline.

## Core Equation

    γ̈ + (1/2) ω_s² sin(2γ) + ṅ + ε γ̇ = 0

where:
- γ = φ - M (spin angle minus mean anomaly)
- ω_s² = 3 n² (B-A)/C |H(e)| (triaxial restoring torque)
- ṅ = mean motion derivative from N-body secular perturbations
- ε = tidal dissipation coefficient (constant time-lag model)

## Key Approximations

1. **One-way coupling**: Orbit affects spin, but spin doesn't affect orbit.
   Valid for timescales << tidal orbital decay timescale (~Gyr).

2. **Constant ε**: Tidal dissipation computed once from median orbital
   parameters. Valid when a(t) and n(t) variations are small (<1%).

3. **Spline-interpolated ṅ**: Mean motion derivative from cubic spline of
   discrete N-body samples. Smooths high-frequency perturbations.

4. **No obliquity**: Assumes spin axis perpendicular to orbital plane.

5. **No thermal tides**: Atmospheric torques not included.

## Validity Scope

- **Timescale**: ~1000 years (limited by N-body integration cost)
- **Eccentricity**: e < 0.3 (system design constraint; H(e) > 0.8)
- **NOT valid for**: Gyr evolution, instability events, tidal orbital decay

## References

- Goldreich & Peale 1966, AJ 71, 425
- Shakespeare & Steffen 2023, ApJ 959, 170
- MacDonald & Dawson 2018, AJ 156, 228
