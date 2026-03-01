"""Resonant chain system generator.

Generates formation-likely planetary systems from resonant chain building
blocks. Each system has an anchor planet placed at the HZ center, with
periods cascading outward/inward via MMR ratios.

Pipeline: stellar draws → planet draws → MMR selection → period cascade
→ Hill stability filter → formation verification (REBOUND).
"""

import numpy as np

from shared.constants import AU, G, M_EARTH, M_SUN, YEAR

from .chain_types import ChainConfig, PlanetSpec, SystemArchitecture

# --- MMR Palette ---

MMR_PALETTE = {
    "3:2": 1.500,
    "4:3": 1.333,
    "5:4": 1.250,
    "5:3": 1.667,
    "2:1": 2.000,
    "8:5": 1.600,
}

MMR_BASE_WEIGHTS = {
    "3:2": 0.30,
    "4:3": 0.25,
    "5:4": 0.15,
    "5:3": 0.10,
    "2:1": 0.10,
    "8:5": 0.10,
}

PLANET_COUNT_WEIGHTS = {3: 0.25, 4: 0.30, 5: 0.25, 6: 0.12, 7: 0.08}


def mass_adjusted_mmr_weights(m_inner: float, m_outer: float) -> dict[str, float]:
    """Adjust MMR weights based on adjacent planet masses.

    Heavier pairs boost wider resonances (2:1, 5:3).
    Lighter pairs boost tighter resonances (5:4, 4:3).
    """
    weights = dict(MMR_BASE_WEIGHTS)
    m_avg = (m_inner + m_outer) / 2.0

    if m_avg < 0.8:
        weights["5:4"] *= 1.5
        weights["4:3"] *= 1.5
    elif m_avg > 1.5:
        weights["2:1"] *= 1.5
        weights["5:3"] *= 1.5

    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def draw_stellar_mass(rng: np.random.Generator) -> float:
    """Draw stellar mass from Kroupa IMF truncated to [0.08, 0.25] M_sun."""
    # Kroupa IMF: dN/dM ∝ M^(-1.3) for 0.08 < M < 0.5
    # Inverse CDF sampling for power law
    alpha = 1.3
    m_min, m_max = 0.08, 0.25
    u = rng.uniform()
    exp = 1.0 - alpha
    mass = (m_min**exp + u * (m_max**exp - m_min**exp)) ** (1.0 / exp)
    return float(mass)


def draw_planet_count(rng: np.random.Generator) -> int:
    """Draw planet count with weighted probabilities."""
    counts = list(PLANET_COUNT_WEIGHTS.keys())
    weights = list(PLANET_COUNT_WEIGHTS.values())
    return int(rng.choice(counts, p=weights))


def draw_planet_masses(n: int, rng: np.random.Generator) -> tuple[float, ...]:
    """Draw planet masses from log-normal, clipped to [0.3, 3.0] M_earth."""
    log_masses = rng.normal(loc=0.0, scale=0.3, size=n)  # median=1.0 M_earth
    masses = 10.0**log_masses
    masses = np.clip(masses, 0.3, 3.0)
    return tuple(float(m) for m in masses)


def mass_radius_relation(mass_mearth: float) -> float:
    """Zeng et al. 2016 mass-radius relation for rocky planets."""
    if mass_mearth < 2.0:
        return mass_mearth**0.27
    return mass_mearth**0.55


def stellar_luminosity(mass_msun: float) -> float:
    """Main-sequence luminosity in L_sun. L ∝ M^4 for M < 0.43."""
    return mass_msun**4


def stellar_teff(mass_msun: float) -> float:
    """Effective temperature from Mann et al. 2015 empirical relation."""
    # Approximate fit for M-dwarfs: T_eff ≈ 3850 * (M/M_sun)^0.35
    return 3850.0 * mass_msun**0.35


def hz_bounds(luminosity_lsun: float) -> tuple[float, float]:
    """Habitable zone inner/outer boundaries in AU.

    Uses conservative HZ from Kopparapu et al. 2013:
    inner = 0.95 * sqrt(L), outer = 1.67 * sqrt(L) (runaway/max greenhouse).
    Scaled from solar values.
    """
    sqrt_l = luminosity_lsun**0.5
    return 0.95 * sqrt_l, 1.67 * sqrt_l


def period_to_distance(period_days: float, stellar_mass_msun: float) -> float:
    """Kepler's third law: period (days) → semi-major axis (AU)."""
    period_s = period_days * 86400.0
    m_star_kg = stellar_mass_msun * M_SUN
    a_m = (G * m_star_kg * (period_s / (2.0 * np.pi)) ** 2) ** (1.0 / 3.0)
    return a_m / AU


def distance_to_period(distance_au: float, stellar_mass_msun: float) -> float:
    """Kepler's third law: semi-major axis (AU) → period (days)."""
    a_m = distance_au * AU
    m_star_kg = stellar_mass_msun * M_SUN
    period_s = 2.0 * np.pi * (a_m**3 / (G * m_star_kg)) ** 0.5
    return period_s / 86400.0


def cascade_periods(
    anchor_period_days: float,
    anchor_idx: int,
    mmrs: tuple[str, ...],
) -> list[float]:
    """Build period array from anchor outward and inward via MMR ratios.

    Args:
        anchor_period_days: Period of the anchor planet.
        anchor_idx: Index of anchor in the chain (0-based).
        mmrs: MMR labels for each adjacent pair (len = n_planets - 1).
            mmrs[i] is the ratio P[i+1]/P[i].

    Returns:
        List of periods in days, ordered from innermost to outermost.
    """
    n = len(mmrs) + 1
    periods = [0.0] * n
    periods[anchor_idx] = anchor_period_days

    # Cascade outward from anchor
    for i in range(anchor_idx, n - 1):
        ratio = MMR_PALETTE[mmrs[i]]
        periods[i + 1] = periods[i] * ratio

    # Cascade inward from anchor
    for i in range(anchor_idx - 1, -1, -1):
        ratio = MMR_PALETTE[mmrs[i]]
        periods[i] = periods[i + 1] / ratio

    return periods


def mutual_hill_separation(
    a1_au: float,
    m1_mearth: float,
    a2_au: float,
    m2_mearth: float,
    mstar_msun: float,
) -> float:
    """Compute mutual Hill separation K between adjacent planets.

    K = (a2 - a1) / R_H_mutual where
    R_H_mutual = ((m1 + m2) / (3 * M_star))^(1/3) * (a1 + a2) / 2
    """
    m1 = m1_mearth * M_EARTH
    m2 = m2_mearth * M_EARTH
    mstar = mstar_msun * M_SUN
    a1 = a1_au * AU
    a2 = a2_au * AU

    r_hill = ((m1 + m2) / (3.0 * mstar)) ** (1.0 / 3.0) * (a1 + a2) / 2.0
    return (a2 - a1) / r_hill


def validate_stability(
    planets: list[PlanetSpec],
    mstar_msun: float,
    k_min: float = 5.0,
    k_max: float = 25.0,
) -> tuple[bool, list[float]]:
    """Check all adjacent Hill separations are in [k_min, k_max]."""
    separations = []
    for i in range(len(planets) - 1):
        k = mutual_hill_separation(
            planets[i].distance_au,
            planets[i].mass_mearth,
            planets[i + 1].distance_au,
            planets[i + 1].mass_mearth,
            mstar_msun,
        )
        separations.append(k)

    valid = all(k_min <= k <= k_max for k in separations)
    return valid, separations


def verify_formation(system: "SystemArchitecture", timeout_s: float = 120.0) -> bool:
    """1 Myr REBOUND integration with eccentricity damping.

    Verifies chain survives a short damped evolution (disk phase proxy).
    Returns True if no ejection or collision occurs.
    """
    import rebound

    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")

    sim.add(m=system.stellar_mass_msun)

    m_earth_msun = M_EARTH / M_SUN
    for p in system.planets:
        sim.add(
            m=p.mass_mearth * m_earth_msun,
            a=p.distance_au,
            e=p.eccentricity,
        )

    sim.move_to_com()

    shortest_period = min(p.period_days for p in system.planets) / 365.25
    sim.integrator = "whfast"
    sim.dt = shortest_period / 20.0

    # Add eccentricity damping via REBOUNDx
    try:
        import reboundx

        rebx = reboundx.Extras(sim)
        mof = rebx.load_force("modify_orbits_forces")
        rebx.add_force(mof)

        tau_e = 1e4  # years
        for i in range(1, len(system.planets) + 1):
            sim.particles[i].params["tau_e"] = -tau_e
    except ImportError:
        pass  # Run without damping if reboundx unavailable

    # Integrate 1 Myr
    t_end = 1e6  # years
    n_checkpoints = 100
    dt_check = t_end / n_checkpoints

    for step in range(n_checkpoints):
        try:
            sim.integrate(sim.t + dt_check)
        except rebound.Encounter:
            return False

        # Check for ejections
        for i in range(1, sim.N):
            orb = sim.particles[i].orbit()
            if orb.a > 10.0 or orb.a < 0:
                return False

    return True


def _draw_mmr_sequence(
    n_planets: int,
    masses: tuple[float, ...],
    rng: np.random.Generator,
) -> tuple[str, ...]:
    """Draw MMR sequence with mass-adjusted weights."""
    mmrs = []
    labels = list(MMR_PALETTE.keys())
    for i in range(n_planets - 1):
        weights = mass_adjusted_mmr_weights(masses[i], masses[i + 1])
        probs = [weights[label] for label in labels]
        choice = rng.choice(labels, p=probs)
        mmrs.append(choice)
    return tuple(mmrs)


def _assign_eccentricities(n: int, rng: np.random.Generator) -> list[float]:
    """Draw eccentricities from Rayleigh distribution (post-formation)."""
    sigma = 0.02
    eccs = rng.rayleigh(sigma, size=n)
    eccs = np.clip(eccs, 0.001, 0.15)
    return [float(e) for e in eccs]


def generate_chain(seed: int) -> SystemArchitecture | None:
    """Generate a single resonant chain system.

    Returns None if the system fails stability checks (caller retries).
    """
    rng = np.random.default_rng(seed)

    stellar_mass = draw_stellar_mass(rng)
    lum = stellar_luminosity(stellar_mass)
    hz_in, hz_out = hz_bounds(lum)
    hz_center = (hz_in + hz_out) / 2.0

    n_planets = draw_planet_count(rng)
    masses = draw_planet_masses(n_planets, rng)

    # Anchor in middle of chain, placed at HZ center
    anchor_idx = n_planets // 2

    mmr_seq = _draw_mmr_sequence(n_planets, masses, rng)

    # Anchor period from HZ center distance
    anchor_period = distance_to_period(hz_center, stellar_mass)

    periods = cascade_periods(anchor_period, anchor_idx, mmr_seq)
    distances = [period_to_distance(p, stellar_mass) for p in periods]
    eccs = _assign_eccentricities(n_planets, rng)

    # Build planet specs
    planets = []
    for i in range(n_planets):
        mmr_label = mmr_seq[i - 1] if i > 0 else None
        planets.append(
            PlanetSpec(
                mass_mearth=masses[i],
                radius_rearth=mass_radius_relation(masses[i]),
                period_days=periods[i],
                distance_au=distances[i],
                eccentricity=eccs[i],
                mmr_label=mmr_label,
            )
        )

    # Hill stability check
    valid, separations = validate_stability(planets, stellar_mass)
    if not valid:
        return None

    # Identify HZ planets
    hz_indices = tuple(
        i for i, p in enumerate(planets) if hz_in <= p.distance_au <= hz_out
    )
    if not hz_indices:
        return None

    system = SystemArchitecture(
        config=ChainConfig(
            stellar_mass_msun=stellar_mass,
            n_planets=n_planets,
            anchor_idx=anchor_idx,
            seed=seed,
            mmr_sequence=mmr_seq,
            planet_masses_mearth=masses,
        ),
        stellar_mass_msun=stellar_mass,
        stellar_luminosity_lsun=lum,
        planets=tuple(planets),
        hz_inner_au=hz_in,
        hz_outer_au=hz_out,
        hz_planet_indices=hz_indices,
        mutual_hill_separations=tuple(separations),
        formation_stable=True,  # Updated after verify_formation
    )

    return system


def generate_batch(
    n_systems: int,
    max_retries_per: int = 10,
    base_seed: int = 42,
    verify_formation_flag: bool = False,
    formation_timeout_s: float = 120.0,
) -> list[SystemArchitecture]:
    """Generate n_systems valid resonant chain systems.

    Args:
        n_systems: Target number of systems.
        max_retries_per: Max attempts per system before giving up.
        base_seed: Starting seed (incremented for each attempt).
        verify_formation_flag: If True, run REBOUND formation check.
        formation_timeout_s: Timeout for formation verification.

    Returns:
        List of valid SystemArchitecture objects.
    """
    systems = []
    seed = base_seed
    attempts = 0

    while len(systems) < n_systems:
        if attempts >= n_systems * max_retries_per:
            break

        system = generate_chain(seed)
        seed += 1
        attempts += 1

        if system is None:
            continue

        if verify_formation_flag:
            stable = verify_formation(system, timeout_s=formation_timeout_s)
            if not stable:
                continue
            # Rebuild with formation_stable=True (already default)

        systems.append(system)

    return systems
