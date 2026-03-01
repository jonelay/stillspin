"""Type-safe, validated configuration and result objects for parameter sweeps.

Addresses v3.2 infrastructure failures:
- Dict-based configs with no validation -> early rejection of invalid params
- No deterministic IDs -> deduplication for resume support

v3.3 additions:
- Episode dataclass for temporal pattern analysis
- SweepResult.episodes field (optional, backward compatible)
- Schema versioning for future migrations

v3.5 additions:
- SweepConfig.stellar_mass_msun for multi-star sweeps (default 0.15)
- Backward compatible: old configs without stellar_mass default to 0.15
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
import hashlib


@dataclass(frozen=True)
class Episode:
    """A single contiguous spin-orbit regime episode.

    Extracted from regime_classifier.Regime for serialization.

    Fields:
        type: Regime type ("TL_ZERO", "TL_PI", "SPINNING", "PTB")
        duration_yr: Episode duration in years
        t_start_yr: Start time in years
        t_end_yr: End time in years
        neighbors: Optional dict with before/after regime types (PTB only)
    """

    type: str
    duration_yr: float
    t_start_yr: float
    t_end_yr: float
    neighbors: dict[str, str | None] | None = None

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        d = {
            "type": self.type,
            "duration_yr": self.duration_yr,
            "t_start_yr": self.t_start_yr,
            "t_end_yr": self.t_end_yr,
        }
        if self.neighbors is not None:
            d["neighbors"] = self.neighbors
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Episode":
        """Create from dict (for loading from JSON)."""
        return cls(
            type=d["type"],
            duration_yr=d["duration_yr"],
            t_start_yr=d["t_start_yr"],
            t_end_yr=d["t_end_yr"],
            neighbors=d.get("neighbors"),
        )


@dataclass(frozen=True)
class SweepConfig:
    """Immutable, validated sweep configuration.

    Physical bounds are validated in __post_init__ to reject invalid
    configs before they reach expensive N-body simulation.

    Fields:
        stellar_mass_msun: Stellar mass in solar masses (default 0.15 for M5.5V).
            Used for HZ distance scaling and orbital period calculation.
            Valid range: [0.08, 0.80] (M8V to K5V).
    """

    q: int
    distance_au: float
    triax: float
    albedo: float = 0.35
    co2_pct: float = 0.6
    n_orbits: int = 20_000
    seed: int | None = None
    stellar_mass_msun: float = 0.15  # Default: M5.5V (Bipolaris primary)

    def __post_init__(self):
        if not (5 <= self.q <= 1000):
            raise ValueError(f"Q={self.q} outside physical range [5, 1000]")
        if not (0.08 <= self.stellar_mass_msun <= 0.80):
            raise ValueError(f"stellar_mass={self.stellar_mass_msun} outside range [0.08, 0.80]")
        # Distance validation scales with stellar mass (HZ scales as M^2)
        # For M=0.08: HZ ~0.017-0.035 AU; for M=0.80: HZ ~1.7-3.5 AU
        # Use broad range [0.01, 0.50] to cover M-dwarf to late K-dwarf HZ
        if not (0.01 <= self.distance_au <= 0.50):
            raise ValueError(f"distance={self.distance_au} outside range [0.01, 0.50]")
        if not (1e-6 <= self.triax <= 1e-2):
            raise ValueError(f"triax={self.triax} outside range [1e-6, 1e-2]")

    @property
    def config_id(self) -> str:
        """Deterministic hash for deduplication and resume."""
        key = (
            f"{self.q}_{self.distance_au:.6f}_{self.triax:.2e}_"
            f"{self.albedo}_{self.co2_pct}_{self.stellar_mass_msun:.4f}"
        )
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            "q": self.q,
            "distance_au": self.distance_au,
            "triax": self.triax,
            "albedo": self.albedo,
            "co2_pct": self.co2_pct,
            "n_orbits": self.n_orbits,
            "seed": self.seed,
            "stellar_mass_msun": self.stellar_mass_msun,
            "config_id": self.config_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SweepConfig":
        """Create from dict (for loading from JSON).

        Backward compatible: configs without stellar_mass_msun default to 0.15.
        """
        return cls(
            q=d["q"],
            distance_au=d["distance_au"],
            triax=d["triax"],
            albedo=d.get("albedo", 0.35),
            co2_pct=d.get("co2_pct", 0.6),
            n_orbits=d.get("n_orbits", 20_000),
            seed=d.get("seed"),
            stellar_mass_msun=d.get("stellar_mass_msun", 0.15),
        )


Status = Literal["OK", "TIMEOUT", "ERROR"]


@dataclass
class SweepResult:
    """Result from a single sweep configuration run.

    Status enum makes aggregation logic explicit:
    - OK: Completed successfully, all fields populated
    - TIMEOUT: Exceeded timeout, partial data may be available
    - ERROR: Exception occurred, error_msg populated

    Schema Evolution:
    - v3.3+: Added episodes field for temporal pattern analysis
    - v3.3+: Added schema_version field for backward compatibility tracking

    Fields:
        episodes: Optional list of Episode objects with temporal sequence data.
            Includes individual episode durations, regime transitions, and neighbors
            for pattern analysis (e.g., TL0 ↔ TLπ flip-flops via PTB).
    """

    config: SweepConfig
    status: Status
    fractions: dict | None
    temps: dict | None
    quality: dict | None
    elapsed_s: float
    episodes: list[Episode] | None = None
    error_msg: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    schema_version: str = "v3.5"

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        d = {
            "config": self.config.to_dict(),
            "status": self.status,
            "fractions": self.fractions,
            "temps": self.temps,
            "quality": self.quality,
            "elapsed_s": self.elapsed_s,
            "error_msg": self.error_msg,
            "timestamp": self.timestamp,
            "schema_version": self.schema_version,
        }
        if self.episodes is not None:
            d["episodes"] = [ep.to_dict() for ep in self.episodes]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SweepResult":
        """Create from dict (for loading from JSONL).

        Backward compatible: handles results without episodes or schema_version.
        """
        episodes = None
        if "episodes" in d and d["episodes"] is not None:
            episodes = [Episode.from_dict(ep) for ep in d["episodes"]]

        return cls(
            config=SweepConfig.from_dict(d["config"]),
            status=d["status"],
            fractions=d.get("fractions"),
            temps=d.get("temps"),
            quality=d.get("quality"),
            elapsed_s=d["elapsed_s"],
            episodes=episodes,
            error_msg=d.get("error_msg"),
            timestamp=d.get("timestamp", ""),
            schema_version=d.get("schema_version", "v3.2"),  # default for old results
        )
