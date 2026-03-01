"""Type-safe dataclasses for the resonant chain survey pipeline.

Follows the SweepConfig/SweepResult pattern from shared/sweep_types.py:
frozen where immutable, validated in __post_init__, deterministic config_id
via MD5, to_dict()/from_dict() for JSONL serialization.

Linking key: All pipeline stages use system_id (= ChainConfig.config_id).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
import hashlib


@dataclass(frozen=True)
class PlanetSpec:
    """Specification for a single planet in a resonant chain."""

    mass_mearth: float
    radius_rearth: float
    period_days: float
    distance_au: float
    eccentricity: float
    mmr_label: str | None = None

    def __post_init__(self):
        if not (0.1 <= self.mass_mearth <= 10.0):
            raise ValueError(f"mass={self.mass_mearth} outside range [0.1, 10.0] M_earth")
        if not (0.001 <= self.distance_au <= 1.0):
            raise ValueError(f"distance={self.distance_au} outside range [0.001, 1.0] AU")
        if not (0.0 <= self.eccentricity <= 0.3):
            raise ValueError(f"eccentricity={self.eccentricity} outside range [0.0, 0.3]")

    def to_dict(self) -> dict:
        return {
            "mass_mearth": self.mass_mearth,
            "radius_rearth": self.radius_rearth,
            "period_days": self.period_days,
            "distance_au": self.distance_au,
            "eccentricity": self.eccentricity,
            "mmr_label": self.mmr_label,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PlanetSpec":
        return cls(**d)


@dataclass(frozen=True)
class ChainConfig:
    """Generation seed for a resonant chain system.

    Everything needed to deterministically reproduce a system.
    """

    stellar_mass_msun: float
    n_planets: int
    anchor_idx: int
    seed: int
    mmr_sequence: tuple[str, ...]
    planet_masses_mearth: tuple[float, ...]

    def __post_init__(self):
        if not (0.08 <= self.stellar_mass_msun <= 0.25):
            raise ValueError(
                f"stellar_mass={self.stellar_mass_msun} outside range [0.08, 0.25]"
            )
        if not (3 <= self.n_planets <= 7):
            raise ValueError(f"n_planets={self.n_planets} outside range [3, 7]")
        if not (0 <= self.anchor_idx < self.n_planets):
            raise ValueError(
                f"anchor_idx={self.anchor_idx} out of range [0, {self.n_planets})"
            )
        if len(self.mmr_sequence) != self.n_planets - 1:
            raise ValueError(
                f"mmr_sequence length {len(self.mmr_sequence)} != n_planets-1 ({self.n_planets - 1})"
            )
        if len(self.planet_masses_mearth) != self.n_planets:
            raise ValueError(
                f"planet_masses length {len(self.planet_masses_mearth)} != n_planets ({self.n_planets})"
            )

    @property
    def config_id(self) -> str:
        key = (
            f"{self.stellar_mass_msun:.4f}_{self.n_planets}_{self.anchor_idx}_"
            f"{self.seed}_{self.mmr_sequence}_{self.planet_masses_mearth}"
        )
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        return {
            "stellar_mass_msun": self.stellar_mass_msun,
            "n_planets": self.n_planets,
            "anchor_idx": self.anchor_idx,
            "seed": self.seed,
            "mmr_sequence": list(self.mmr_sequence),
            "planet_masses_mearth": list(self.planet_masses_mearth),
            "config_id": self.config_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ChainConfig":
        return cls(
            stellar_mass_msun=d["stellar_mass_msun"],
            n_planets=d["n_planets"],
            anchor_idx=d["anchor_idx"],
            seed=d["seed"],
            mmr_sequence=tuple(d["mmr_sequence"]),
            planet_masses_mearth=tuple(d["planet_masses_mearth"]),
        )


@dataclass(frozen=True)
class SystemArchitecture:
    """Fully realized system after chain construction + stability filter."""

    config: ChainConfig
    stellar_mass_msun: float
    stellar_luminosity_lsun: float
    planets: tuple[PlanetSpec, ...]
    hz_inner_au: float
    hz_outer_au: float
    hz_planet_indices: tuple[int, ...]
    mutual_hill_separations: tuple[float, ...]
    formation_stable: bool

    @property
    def system_id(self) -> str:
        return self.config.config_id

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "stellar_mass_msun": self.stellar_mass_msun,
            "stellar_luminosity_lsun": self.stellar_luminosity_lsun,
            "planets": [p.to_dict() for p in self.planets],
            "hz_inner_au": self.hz_inner_au,
            "hz_outer_au": self.hz_outer_au,
            "hz_planet_indices": list(self.hz_planet_indices),
            "mutual_hill_separations": list(self.mutual_hill_separations),
            "formation_stable": self.formation_stable,
            "system_id": self.system_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SystemArchitecture":
        return cls(
            config=ChainConfig.from_dict(d["config"]),
            stellar_mass_msun=d["stellar_mass_msun"],
            stellar_luminosity_lsun=d["stellar_luminosity_lsun"],
            planets=tuple(PlanetSpec.from_dict(p) for p in d["planets"]),
            hz_inner_au=d["hz_inner_au"],
            hz_outer_au=d["hz_outer_au"],
            hz_planet_indices=tuple(d["hz_planet_indices"]),
            mutual_hill_separations=tuple(d["mutual_hill_separations"]),
            formation_stable=d["formation_stable"],
        )


EvolutionStatus = Literal[
    "INTACT", "PARTIAL_BREAK", "FULL_BREAK", "EJECTION", "COLLISION", "TIMEOUT", "ERROR"
]


@dataclass
class OrbitalResult:
    """Pipeline 1 output: orbital evolution result."""

    system_id: str
    status: EvolutionStatus
    final_planets: list[dict]
    breaking_events: list[dict]
    n_survivors: int
    hz_planet_survived: bool
    hz_planet_idx: int
    elapsed_s: float
    error_msg: str | None = None
    sim_archive_path: str | None = None

    def to_dict(self) -> dict:
        return {
            "system_id": self.system_id,
            "status": self.status,
            "final_planets": self.final_planets,
            "breaking_events": self.breaking_events,
            "n_survivors": self.n_survivors,
            "hz_planet_survived": self.hz_planet_survived,
            "hz_planet_idx": self.hz_planet_idx,
            "elapsed_s": self.elapsed_s,
            "error_msg": self.error_msg,
            "sim_archive_path": self.sim_archive_path,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OrbitalResult":
        return cls(
            system_id=d["system_id"],
            status=d["status"],
            final_planets=d["final_planets"],
            breaking_events=d["breaking_events"],
            n_survivors=d["n_survivors"],
            hz_planet_survived=d["hz_planet_survived"],
            hz_planet_idx=d["hz_planet_idx"],
            elapsed_s=d["elapsed_s"],
            error_msg=d.get("error_msg"),
            sim_archive_path=d.get("sim_archive_path"),
        )


FilterVerdict = Literal["PASS", "REJECT_WEAK", "REJECT_STRONG", "ERROR"]


@dataclass
class PerturbationProbe:
    """N-body probe output (5K years)."""

    system_id: str
    hz_planet_idx: int
    rms_dn_over_n: float
    rms_de: float
    max_e: float
    mean_n_rad_s: float
    dominant_periods_yr: list[float]
    filter_verdict: FilterVerdict
    elapsed_s: float
    error_msg: str | None = None

    def to_dict(self) -> dict:
        return {
            "system_id": self.system_id,
            "hz_planet_idx": self.hz_planet_idx,
            "rms_dn_over_n": self.rms_dn_over_n,
            "rms_de": self.rms_de,
            "max_e": self.max_e,
            "mean_n_rad_s": self.mean_n_rad_s,
            "dominant_periods_yr": self.dominant_periods_yr,
            "filter_verdict": self.filter_verdict,
            "elapsed_s": self.elapsed_s,
            "error_msg": self.error_msg,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PerturbationProbe":
        return cls(
            system_id=d["system_id"],
            hz_planet_idx=d["hz_planet_idx"],
            rms_dn_over_n=d["rms_dn_over_n"],
            rms_de=d["rms_de"],
            max_e=d["max_e"],
            mean_n_rad_s=d["mean_n_rad_s"],
            dominant_periods_yr=d["dominant_periods_yr"],
            filter_verdict=d["filter_verdict"],
            elapsed_s=d["elapsed_s"],
            error_msg=d.get("error_msg"),
        )


SpinStatus = Literal["OK", "TIMEOUT", "ERROR"]


@dataclass
class SpinSurveyResult:
    """Pipeline 2 output: spin dynamics result."""

    system_id: str
    tidal_q: int
    triaxiality: float
    fractions: dict | None
    is_flipflop: bool
    episodes: list[dict] | None
    status: SpinStatus
    elapsed_s: float
    error_msg: str | None = None

    def to_dict(self) -> dict:
        return {
            "system_id": self.system_id,
            "tidal_q": self.tidal_q,
            "triaxiality": self.triaxiality,
            "fractions": self.fractions,
            "is_flipflop": self.is_flipflop,
            "episodes": self.episodes,
            "status": self.status,
            "elapsed_s": self.elapsed_s,
            "error_msg": self.error_msg,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SpinSurveyResult":
        return cls(
            system_id=d["system_id"],
            tidal_q=d["tidal_q"],
            triaxiality=d["triaxiality"],
            fractions=d.get("fractions"),
            is_flipflop=d["is_flipflop"],
            episodes=d.get("episodes"),
            status=d["status"],
            elapsed_s=d["elapsed_s"],
            error_msg=d.get("error_msg"),
        )
