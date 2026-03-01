"""Analysis utilities for parameter sensitivity and viability scoring.

Used by scripts/sensitivity_analysis.py to quantify credibility and rarity
of the Bipolaris v3.2 configuration.

v3.4 additions:
- compute_episode_statistics: Extract temporal pattern metrics
- compute_surface_conditions: Habitability metrics
- compute_slow_bouncer_score: Combined score for slow bouncer worlds
- filter_slow_bouncer_candidates: Filter and rank candidates

v3.5 additions:
- hz_distance_for_mass: Compute HZ boundaries for different stellar masses
"""

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

if TYPE_CHECKING:
    from shared.sweep_types import Episode, SweepConfig, SweepResult


def classify_config_risk(config: "SweepConfig") -> Literal["safe", "moderate", "risky"]:
    """Classify config by convergence risk based on v3.2 findings.

    Used to sort configs before execution: safe first, risky last with
    extended timeout. Risky configs caused the 23-hour hang in v3.2.

    Risk factors:
    - Q < 10: Very low Q causes instability
    - Q > 50: High Q with high triax can be slow
    - Triax far from 3e-5: Far from known-good baseline
    - Distance far from resonance centers: Outside stable regions

    Args:
        config: SweepConfig to classify.

    Returns:
        "safe", "moderate", or "risky"
    """
    risk = 0

    if config.q < 10:
        risk += 2
    if config.q > 50:
        risk += 1
    if abs(config.triax - 3e-5) > 1.5e-5:
        risk += 1

    # Empirical PTB peaks from v3.2 resonance sweeps
    resonance_centers = [0.0675, 0.0695, 0.072, 0.0735]
    min_dist_from_resonance = min(abs(config.distance_au - c) for c in resonance_centers)
    if min_dist_from_resonance > 0.003:
        risk += 1

    if risk >= 3:
        return "risky"
    elif risk >= 1:
        return "moderate"
    return "safe"


def hz_distance_for_mass(stellar_mass_msun: float) -> tuple[float, float, float]:
    """Compute habitable zone boundaries for a given stellar mass.

    Uses L ∝ M^4 approximation for main-sequence M-dwarfs (valid for
    M_star < 0.43 M_sun). HZ scales as sqrt(L), so HZ ∝ M^2.

    For Bipolaris baseline (0.15 M_sun):
    - L ≈ 0.0034 L_sun
    - HZ_inner ≈ 0.048 AU, HZ_outer ≈ 0.098 AU

    Args:
        stellar_mass_msun: Stellar mass in solar masses [0.08, 0.80].

    Returns:
        Tuple of (hz_inner_au, hz_center_au, hz_outer_au).

    Example:
        >>> hz_inner, hz_center, hz_outer = hz_distance_for_mass(0.15)
        >>> print(f"HZ: {hz_inner:.3f} - {hz_outer:.3f} AU")
        HZ: 0.048 - 0.098 AU
    """
    # Reference: Bipolaris at 0.15 M_sun
    m_ref = 0.15
    hz_inner_ref = 0.048
    hz_outer_ref = 0.098

    # L ∝ M^4 for low-mass stars, HZ ∝ sqrt(L) ∝ M^2
    scale = (stellar_mass_msun / m_ref) ** 2

    hz_inner = hz_inner_ref * scale
    hz_outer = hz_outer_ref * scale
    hz_center = (hz_inner + hz_outer) / 2

    return hz_inner, hz_center, hz_outer


def compute_ptb_quality_score(
    ptb_frac: float,
    tl_zero_frac: float,
    tl_pi_frac: float,
    t_term: float,
    target_ptb: float = 0.36,
    target_tl_zero: float = 0.30,
    target_tl_pi: float = 0.34,
    t_term_min: float = 260.0,
    t_term_max: float = 290.0,
) -> dict:
    """Compute viability score for a configuration.

    Combines PTB regime balance with thermal habitability to produce
    a single quality metric for parameter sweeps.

    Args:
        ptb_frac: PTB regime fraction (0-1).
        tl_zero_frac: TL_ZERO regime fraction (0-1).
        tl_pi_frac: TL_PI regime fraction (0-1).
        t_term: Terminator temperature (K).
        target_ptb: Target PTB fraction (default: v3.2 baseline).
        target_tl_zero: Target TL_ZERO fraction.
        target_tl_pi: Target TL_PI fraction.
        t_term_min: Minimum viable terminator temperature (K).
        t_term_max: Maximum viable terminator temperature (K).

    Returns:
        Dict with:
        - regime_score: Distance from target regime balance (0-1, higher is better)
        - thermal_score: Temperature viability (0-1, 1 = in range)
        - overall_score: Combined score (product of regime and thermal)
        - is_viable: True if PTB > 0.1 and t_term in range
    """
    # Regime balance: Euclidean distance from target
    ptb_diff = (ptb_frac - target_ptb) ** 2
    tl0_diff = (tl_zero_frac - target_tl_zero) ** 2
    tlpi_diff = (tl_pi_frac - target_tl_pi) ** 2
    regime_dist = np.sqrt(ptb_diff + tl0_diff + tlpi_diff)
    regime_score = np.exp(-3 * regime_dist)  # Exponential decay

    # Thermal: inside range = 1, linear decay outside
    if t_term_min <= t_term <= t_term_max:
        thermal_score = 1.0
    elif t_term < t_term_min:
        thermal_score = max(0, 1 - (t_term_min - t_term) / 50)
    else:
        thermal_score = max(0, 1 - (t_term - t_term_max) / 50)

    # Overall: product (both must be good)
    overall_score = regime_score * thermal_score

    # Viability threshold
    is_viable = ptb_frac > 0.1 and t_term_min <= t_term <= t_term_max

    return {
        "regime_score": regime_score,
        "thermal_score": thermal_score,
        "overall_score": overall_score,
        "is_viable": is_viable,
        "regime_distance": regime_dist,
    }


def compute_regime_entropy(fractions: dict) -> float:
    """Compute regime balance entropy (Shannon entropy).

    High entropy = balanced regimes, low entropy = dominated by one regime.

    Args:
        fractions: Dict with regime fractions (keys: PTB, TL_ZERO, TL_PI, SPINNING).

    Returns:
        Shannon entropy in bits (0 = single regime, log2(n) = uniform).
    """
    probs = [fractions.get(k, 0) for k in ["PTB", "TL_ZERO", "TL_PI", "SPINNING"]]
    probs = [p for p in probs if p > 0]  # Filter zeros
    if not probs:
        return 0.0
    return -np.sum([p * np.log2(p) for p in probs])


def estimate_rarity_probability(
    scores: list[float],
    threshold: float,
    method: str = "kde",
) -> dict:
    """Estimate probability of viable configuration from Monte Carlo samples.

    Args:
        scores: List of overall_score values from compute_ptb_quality_score.
        threshold: Minimum score for viability (e.g., 0.5).
        method: Estimation method ('kde' or 'empirical').

    Returns:
        Dict with:
        - p_viable: Estimated probability of viable config (0-1)
        - rarity: 1:X odds (1/p_viable)
        - n_viable: Number of viable samples
        - n_total: Total samples
        - ci_lower: 95% confidence interval lower bound
        - ci_upper: 95% confidence interval upper bound
    """
    scores = np.array(scores)
    n_viable = np.sum(scores >= threshold)
    n_total = len(scores)

    if method == "empirical":
        # Direct fraction
        p_viable = n_viable / n_total if n_total > 0 else 0.0
    elif method == "kde":
        # KDE smoothing (handles low counts better)
        if n_viable > 5:
            kde = stats.gaussian_kde(scores)
            p_viable = kde.integrate_box_1d(threshold, np.inf)
        else:
            # Fallback to empirical for low counts
            p_viable = n_viable / n_total if n_total > 0 else 0.0
    else:
        raise ValueError(f"Unknown method: {method}")

    # Wilson score interval for binomial proportion
    if n_total > 0:
        p_hat = n_viable / n_total
        z = 1.96  # 95% CI
        denom = 1 + z**2 / n_total
        center = (p_hat + z**2 / (2 * n_total)) / denom
        margin = z * np.sqrt(p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2)) / denom
        ci_lower = max(0, center - margin)
        ci_upper = min(1, center + margin)
    else:
        ci_lower = ci_upper = 0.0

    rarity = 1 / p_viable if p_viable > 0 else np.inf

    return {
        "p_viable": p_viable,
        "rarity": rarity,
        "n_viable": n_viable,
        "n_total": n_total,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def fit_resonance_profile(
    distances: np.ndarray,
    ptb_fractions: np.ndarray,
    model: str = "gaussian",
) -> dict:
    """Fit functional form to PTB vs distance curve.

    Used to characterize resonance window width and shape.

    Args:
        distances: Orbital distances (AU).
        ptb_fractions: PTB fractions at each distance (0-1).
        model: 'gaussian' or 'lorentzian'.

    Returns:
        Dict with:
        - params: Fitted parameters [amplitude, center, width]
        - fwhm: Full-width at half-maximum (AU)
        - center: Resonance center (AU)
        - r_squared: Goodness of fit
        - func: Fitted function (callable)
    """
    if model == "gaussian":
        def func(x, a, x0, sigma):
            return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
        fwhm_factor = 2.355  # 2*sqrt(2*ln(2))
    elif model == "lorentzian":
        def func(x, a, x0, gamma):
            return a * gamma ** 2 / ((x - x0) ** 2 + gamma ** 2)
        fwhm_factor = 2.0  # 2*gamma
    else:
        raise ValueError(f"Unknown model: {model}")

    # Initial guess
    a0 = np.max(ptb_fractions)
    x0 = distances[np.argmax(ptb_fractions)]
    w0 = 0.002  # ~2 mAU

    try:
        params, _ = curve_fit(func, distances, ptb_fractions, p0=[a0, x0, w0])
        a, x0, width = params
        fwhm = fwhm_factor * abs(width)

        # R²
        residuals = ptb_fractions - func(distances, *params)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((ptb_fractions - np.mean(ptb_fractions)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            "params": params.tolist(),
            "fwhm": fwhm,
            "center": x0,
            "r_squared": r_squared,
            "func": lambda x: func(x, *params),
            "model": model,
        }
    except RuntimeError:
        # Fit failed, return NaN
        return {
            "params": [np.nan, np.nan, np.nan],
            "fwhm": np.nan,
            "center": np.nan,
            "r_squared": np.nan,
            "func": None,
            "model": model,
        }


def compute_episode_statistics(episodes: list["Episode"] | None) -> dict:
    """Extract statistical summary from episode sequence.

    Analyzes temporal patterns to identify slow bouncer worlds:
    - Long stable TL periods with brief PTB transitions
    - Few total episodes (simple dynamics)
    - Balanced TL0/TLπ distribution

    Args:
        episodes: List of Episode objects from SweepResult.

    Returns:
        Dict with:
        - n_episodes: Total episode count
        - n_tl_episodes: Number of tidal lock episodes (TL_ZERO + TL_PI)
        - n_ptb_episodes: Number of PTB episodes
        - mean_tl_duration_yr: Average TL episode duration
        - max_tl_duration_yr: Longest TL episode
        - mean_ptb_duration_yr: Average PTB episode duration
        - max_ptb_duration_yr: Longest PTB episode (cascade indicator)
        - ptb_chain_fraction: Fraction of time in PTB chains >3 episodes
    """
    if not episodes:
        return {
            "n_episodes": 0,
            "n_tl_episodes": 0,
            "n_ptb_episodes": 0,
            "mean_tl_duration_yr": 0.0,
            "max_tl_duration_yr": 0.0,
            "mean_ptb_duration_yr": 0.0,
            "max_ptb_duration_yr": 0.0,
            "ptb_chain_fraction": 0.0,
        }

    tl_episodes = [ep for ep in episodes if ep.type in ("TL_ZERO", "TL_PI")]
    ptb_episodes = [ep for ep in episodes if ep.type == "PTB"]

    tl_durations = [ep.duration_yr for ep in tl_episodes]
    ptb_durations = [ep.duration_yr for ep in ptb_episodes]

    # Identify PTB chains (consecutive PTB episodes)
    # A chain is 3+ consecutive PTB episodes without stable TL
    chain_time = 0.0
    total_time = sum(ep.duration_yr for ep in episodes)
    i = 0
    while i < len(episodes):
        if episodes[i].type == "PTB":
            chain_start = i
            chain_duration = 0.0
            while i < len(episodes) and episodes[i].type in ("PTB", "SPINNING"):
                chain_duration += episodes[i].duration_yr
                i += 1
            chain_length = i - chain_start
            if chain_length >= 3:
                chain_time += chain_duration
        else:
            i += 1

    return {
        "n_episodes": len(episodes),
        "n_tl_episodes": len(tl_episodes),
        "n_ptb_episodes": len(ptb_episodes),
        "mean_tl_duration_yr": float(np.mean(tl_durations)) if tl_durations else 0.0,
        "max_tl_duration_yr": float(max(tl_durations)) if tl_durations else 0.0,
        "mean_ptb_duration_yr": float(np.mean(ptb_durations)) if ptb_durations else 0.0,
        "max_ptb_duration_yr": float(max(ptb_durations)) if ptb_durations else 0.0,
        "ptb_chain_fraction": chain_time / total_time if total_time > 0 else 0.0,
    }


def compute_surface_conditions(temps: dict | None) -> dict:
    """Compute habitability metrics from thermal model output.

    Args:
        temps: Dict with temperature fields from thermal model.
            Expected keys: t_sub, t_term, t_anti, t_ptb_avg

    Returns:
        Dict with:
        - habitable_fraction: Estimated fraction of surface in 260-290K range
        - cold_trap_active: True if T_anti < 112K (CO2 condensation)
        - ptb_thermal_stress_k: Temperature swing during PTB (|T_ptb - T_term|)
        - ice_stable: True if permanent ice possible (T_anti < 150K)
        - t_term: Terminator temperature (K)
    """
    if not temps:
        return {
            "habitable_fraction": 0.0,
            "cold_trap_active": False,
            "ptb_thermal_stress_k": 0.0,
            "ice_stable": False,
            "t_term": 0.0,
        }

    t_term = temps.get("t_term", 270)
    t_anti = temps.get("t_anti", 120)
    t_ptb_avg = temps.get("t_ptb_avg", t_term)

    # Habitable fraction estimate (simplified: based on terminator width)
    # T_term in 260-290K = 1.0, linear decay outside
    if 260 <= t_term <= 290:
        hab_base = 1.0
    elif t_term < 260:
        hab_base = max(0, 1 - (260 - t_term) / 40)
    else:
        hab_base = max(0, 1 - (t_term - 290) / 40)

    # Scale by terminator region assumption (~20% of surface habitable at best)
    habitable_fraction = hab_base * 0.20

    return {
        "habitable_fraction": habitable_fraction,
        "cold_trap_active": t_anti < 112,
        "ptb_thermal_stress_k": abs(t_ptb_avg - t_term),
        "ice_stable": t_anti < 150,
        "t_term": t_term,
    }


def compute_slow_bouncer_score(
    fractions: dict | None,
    episodes: list["Episode"] | None,
    temps: dict | None,
) -> dict:
    """Compute combined slow bouncer score.

    Scoring weights:
    - 25% TL dominance: High total TL fraction (TL0 + TLπ > 70%)
    - 20% Simplicity: Few episodes (<50 per 1000 yr)
    - 20% Lock duration: Long average TL episodes (>50 yr)
    - 20% Balance: Both TL0 and TLπ present (>15% each)
    - 15% Flip-flop: Some PTB activity (5-30% ideal, penalize 0% or >30%)

    Args:
        fractions: Regime fractions dict.
        episodes: Episode list for temporal analysis.
        temps: Temperature dict for habitability check.

    Returns:
        Dict with:
        - slow_bouncer_score: Combined score (0-1)
        - tl_dominance_score: TL fraction component
        - simplicity_score: Low episode count component
        - lock_duration_score: Long TL episodes component
        - balance_score: TL0/TLπ balance component
        - flipflop_score: PTB activity component (0 for permanent lock)
        - is_slow_bouncer: True if score > 0.5 and habitable
    """
    if not fractions:
        return {
            "slow_bouncer_score": 0.0,
            "tl_dominance_score": 0.0,
            "simplicity_score": 0.0,
            "lock_duration_score": 0.0,
            "balance_score": 0.0,
            "flipflop_score": 0.0,
            "is_slow_bouncer": False,
        }

    tl_zero = fractions.get("TL_ZERO", 0)
    tl_pi = fractions.get("TL_PI", 0)
    ptb = fractions.get("PTB", 0)
    tl_total = tl_zero + tl_pi

    ep_stats = compute_episode_statistics(episodes)
    surface = compute_surface_conditions(temps)

    # TL dominance (25%): want TL > 70%
    # Score 1 at 100%, 0 at 50%
    tl_dominance_score = min(1.0, max(0, (tl_total - 0.50) / 0.50))

    # Simplicity (20%): want < 50 episodes per 1000 yr
    # Assuming 1000 yr simulation
    n_episodes = ep_stats["n_episodes"]
    if n_episodes <= 20:
        simplicity_score = 1.0
    elif n_episodes <= 50:
        simplicity_score = 1.0 - (n_episodes - 20) / 30
    else:
        simplicity_score = max(0, 0.5 - (n_episodes - 50) / 100)

    # Lock duration (20%): want mean TL > 50 yr
    mean_tl = ep_stats["mean_tl_duration_yr"]
    if mean_tl >= 100:
        lock_duration_score = 1.0
    elif mean_tl >= 50:
        lock_duration_score = 0.5 + (mean_tl - 50) / 100
    elif mean_tl >= 20:
        lock_duration_score = (mean_tl - 20) / 60
    else:
        lock_duration_score = 0.0

    # Balance (20%): want both TL0 and TLπ > 15%
    min_tl = min(tl_zero, tl_pi)
    if min_tl >= 0.30:
        balance_score = 1.0
    elif min_tl >= 0.15:
        balance_score = (min_tl - 0.15) / 0.15
    else:
        balance_score = 0.0

    # Flip-flop score (15%): want PTB in 5-30% range
    # 0% PTB = permanent lock = score 0 (not a flip-flop world)
    # 5-25% PTB = ideal range = score 1
    # >30% PTB = too chaotic = score 0
    if ptb < 0.05:
        flipflop_score = ptb / 0.05  # Linear ramp 0->1 from 0% to 5%
    elif ptb <= 0.25:
        flipflop_score = 1.0  # Ideal range
    elif ptb <= 0.30:
        flipflop_score = 1.0 - (ptb - 0.25) / 0.05  # Linear decay
    else:
        flipflop_score = 0.0

    # Combined weighted score
    slow_bouncer_score = (
        0.25 * tl_dominance_score
        + 0.20 * simplicity_score
        + 0.20 * lock_duration_score
        + 0.20 * balance_score
        + 0.15 * flipflop_score
    )

    # Is slow bouncer: score > 0.5, PTB in 5-30%, habitable T_term
    t_term = surface["t_term"]
    is_slow_bouncer = (
        slow_bouncer_score > 0.5
        and 0.05 <= ptb <= 0.30
        and 260 <= t_term <= 290
    )

    return {
        "slow_bouncer_score": slow_bouncer_score,
        "tl_dominance_score": tl_dominance_score,
        "simplicity_score": simplicity_score,
        "lock_duration_score": lock_duration_score,
        "balance_score": balance_score,
        "flipflop_score": flipflop_score,
        "is_slow_bouncer": is_slow_bouncer,
    }


def filter_slow_bouncer_candidates(
    results: list["SweepResult"],
    min_tl_frac: float = 0.70,
    min_ptb_frac: float = 0.05,
    max_ptb_frac: float = 0.30,
    max_episodes: int = 50,
    require_flipflop: bool = True,
) -> list[dict]:
    """Filter sweep results to slow bouncer candidates.

    Args:
        results: List of SweepResult objects.
        min_tl_frac: Minimum total TL fraction (TL0 + TLπ).
        min_ptb_frac: Minimum PTB fraction (filters permanent locks).
        max_ptb_frac: Maximum PTB fraction.
        max_episodes: Maximum episode count.
        require_flipflop: If True, require both TL_ZERO and TL_PI present.

    Returns:
        List of candidate dicts sorted by slow_bouncer_score (descending).
        Each dict has: config, fractions, temps, score_details, ep_stats.
    """
    candidates = []

    for result in results:
        if result.status != "OK" or not result.fractions:
            continue

        fractions = result.fractions
        tl_zero = fractions.get("TL_ZERO", 0)
        tl_pi = fractions.get("TL_PI", 0)
        tl_total = tl_zero + tl_pi
        ptb = fractions.get("PTB", 0)

        # Basic filters
        if tl_total < min_tl_frac:
            continue
        if ptb > max_ptb_frac:
            continue

        # Filter permanent locks (require some PTB activity for flip-flops)
        if ptb < min_ptb_frac:
            continue

        # Require both TL_ZERO and TL_PI for true flip-flop behavior
        if require_flipflop and (tl_zero < 0.01 or tl_pi < 0.01):
            continue

        ep_stats = compute_episode_statistics(result.episodes)
        # Use TL episode count, not total (PTB transitions can be numerous but brief)
        if ep_stats["n_tl_episodes"] > max_episodes:
            continue

        score_details = compute_slow_bouncer_score(
            fractions, result.episodes, result.temps
        )

        candidates.append({
            "config": result.config.to_dict(),
            "fractions": fractions,
            "temps": result.temps,
            "score_details": score_details,
            "ep_stats": ep_stats,
        })

    # Sort by score descending
    candidates.sort(key=lambda c: c["score_details"]["slow_bouncer_score"], reverse=True)

    return candidates
