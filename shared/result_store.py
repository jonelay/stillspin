"""Crash-safe incremental result storage using JSONL.

Addresses v3.2 infrastructure failure: ProcessPoolExecutor only saved
at session end, so crash = all results lost.

Design:
- JSONL (not JSON array): Append-only, no parse-whole-file on each save
- Separate index file: O(1) lookup for completed IDs vs O(n) scan
- No file locking: Single writer (main process), parallel workers return via futures
"""

import json
import logging
from pathlib import Path

from shared.sweep_types import SweepResult


class ResultStore:
    """Single-writer incremental result storage using JSONL."""

    def __init__(self, output_dir: Path | str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.output_dir / "results.jsonl"
        self.index_file = self.output_dir / "index.json"

    def save(self, result: SweepResult) -> None:
        """Append single result immediately (crash-safe)."""
        with open(self.results_file, "a") as f:
            f.write(json.dumps(result.to_dict(), default=str) + "\n")
        self._update_index(result.config.config_id, result.status)

    def _update_index(self, config_id: str, status: str) -> None:
        """Update index file with completed config ID."""
        index = {}
        if self.index_file.exists():
            try:
                index = json.loads(self.index_file.read_text())
            except json.JSONDecodeError:
                pass
        index[config_id] = status
        self.index_file.write_text(json.dumps(index))

    def get_completed_ids(self) -> set[str]:
        """Get IDs of already-completed configs for resume."""
        if not self.index_file.exists():
            return set()
        try:
            return set(json.loads(self.index_file.read_text()).keys())
        except json.JSONDecodeError:
            return set()

    def load_all(self) -> list[dict]:
        """Load all results from JSONL file."""
        if not self.results_file.exists():
            return []
        results = []
        for line in self.results_file.read_text().splitlines():
            if line.strip():
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    logging.warning(f"Skipping malformed JSON line in {self.results_file}")
                    continue
        return results

    def load_results(self) -> list[SweepResult]:
        """Load all results as SweepResult objects."""
        from shared.sweep_types import SweepResult

        return [SweepResult.from_dict(d) for d in self.load_all()]

    def get_stats(self) -> dict:
        """Get summary statistics of stored results."""
        results = self.load_all()
        if not results:
            return {"total": 0, "ok": 0, "timeout": 0, "error": 0}
        return {
            "total": len(results),
            "ok": sum(1 for r in results if r["status"] == "OK"),
            "timeout": sum(1 for r in results if r["status"] == "TIMEOUT"),
            "error": sum(1 for r in results if r["status"] == "ERROR"),
        }

    def mark_pending(self, config_ids: set[str]) -> None:
        """Remove config IDs from index to force re-run.

        Use for re-running specific configs with updated code (e.g., episode extraction).
        Old results remain in JSONL; new results will append.
        """
        if not self.index_file.exists():
            return

        try:
            index = json.loads(self.index_file.read_text())
        except json.JSONDecodeError:
            return

        for config_id in config_ids:
            index.pop(config_id, None)

        self.index_file.write_text(json.dumps(index))
