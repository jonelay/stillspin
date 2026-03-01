#!/usr/bin/env python3
"""CLI orchestrator for the resonant chain survey.

Usage:
  uv run python scripts/run_chain_survey.py --generate 5000
  uv run python scripts/run_chain_survey.py --evolve --workers 24
  uv run python scripts/run_chain_survey.py --calibrate --workers 24
  uv run python scripts/run_chain_survey.py --learn-filter
  uv run python scripts/run_chain_survey.py --probe --workers 24
  uv run python scripts/run_chain_survey.py --spin-survey --workers 24
  uv run python scripts/run_chain_survey.py --all --workers 24
  uv run python scripts/run_chain_survey.py --report

Flags:
  --quick         Reduced samples (50 systems, 0.1 Myr evolution, 1K yr spin)
  --workers N     Parallel workers (default: 1)
  --resume        Skip completed systems
  --output DIR    Output directory (default: results/chain_survey/)
  --timeout N     Per-system timeout in seconds
  --seed N        Base RNG seed (default: 42)
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from chain_survey.chain_generator import generate_batch
from chain_survey.chain_types import (
    OrbitalResult,
    PerturbationProbe,
    SpinSurveyResult,
    SystemArchitecture,
)


def _load_jsonl(path: Path, cls):
    """Load all records from a JSONL file."""
    if not path.exists():
        return []
    results = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                results.append(cls.from_dict(json.loads(line)))
            except (json.JSONDecodeError, Exception):
                continue
    return results


def _json_safe(obj):
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, bool):
        return bool(obj)
    raise TypeError(f"Not serializable: {type(obj)}")


def _save_jsonl(path: Path, record):
    """Append a single record to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record.to_dict(), default=_json_safe) + "\n")


def _load_index(path: Path) -> set[str]:
    """Load completed IDs from index file."""
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text()).keys())
    except (json.JSONDecodeError, Exception):
        return set()


def _update_index(path: Path, record_id: str, status: str):
    """Update index file with completed ID."""
    index = {}
    if path.exists():
        try:
            index = json.loads(path.read_text())
        except (json.JSONDecodeError, Exception):
            pass
    index[record_id] = status
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index))


def cmd_generate(args):
    """Generate resonant chain systems."""
    n = 50 if args.quick else args.generate
    print(f"Generating {n} systems (seed={args.seed})...")

    # Check for REBOUND if formation verification will be needed later
    try:
        import rebound
    except ImportError:
        print("WARNING: REBOUND not installed. Install with: uv sync --extra rebound")
        print("         Chain evolution will fail without REBOUND.")
    t0 = time.time()

    systems = generate_batch(
        n_systems=n,
        max_retries_per=10,
        base_seed=args.seed,
        verify_formation_flag=False,  # Skip formation verify (requires REBOUND)
    )

    out_dir = Path(args.output) / "systems"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / "results.jsonl"
    index_file = out_dir / "index.json"

    # Overwrite if not resuming
    if not args.resume:
        results_file.unlink(missing_ok=True)
        index_file.unlink(missing_ok=True)

    for s in systems:
        _save_jsonl(results_file, s)
        _update_index(index_file, s.system_id, "OK")

    elapsed = time.time() - t0
    print(f"Generated {len(systems)} systems in {elapsed:.1f}s")
    print(f"Output: {results_file}")


class _Heartbeat:
    """Prints periodic status while long-running parallel work is in progress."""

    def __init__(self, total, label="systems", interval=30):
        self.total = total
        self.label = label
        self.interval = interval
        self.done = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def tick(self):
        with self._lock:
            self.done += 1

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self):
        while not self._stop.wait(self.interval):
            with self._lock:
                elapsed = time.time() - self._t0
                mins = int(elapsed // 60)
                secs = int(elapsed % 60)
                print(f"  ... {self.done}/{self.total} {self.label} done "
                      f"({mins}m{secs:02d}s elapsed)", flush=True)


def _evolve_one(args_tuple):
    """Worker function for parallel evolution."""
    system_dict, t_end_myr, output_dir = args_tuple
    from chain_survey.chain_types import SystemArchitecture
    from chain_survey.orbital_evolution import evolve_system
    system = SystemArchitecture.from_dict(system_dict)
    return evolve_system(system, t_end_myr=t_end_myr, output_dir=output_dir)


def cmd_evolve(args):
    """Run Pipeline 1: orbital evolution."""
    systems = _load_jsonl(
        Path(args.output) / "systems" / "results.jsonl", SystemArchitecture,
    )
    if not systems:
        print("No systems found. Run --generate first.")
        return

    out_dir = Path(args.output) / "evolution"
    evo_archive_dir = str(out_dir)
    results_file = out_dir / "results.jsonl"
    index_file = out_dir / "index.json"

    completed = _load_index(index_file) if args.resume else set()
    pending = [s for s in systems if s.system_id not in completed]
    print(f"Evolving {len(pending)}/{len(systems)} systems "
          f"(t_end={0.1 if args.quick else 500.0} Myr, workers={args.workers})")

    t_end = 0.1 if args.quick else 500.0
    work = [(s.to_dict(), t_end, evo_archive_dir) for s in pending]

    t0 = time.time()
    done = 0

    if args.workers <= 1:
        for w in work:
            result = _evolve_one(w)
            _save_jsonl(results_file, result)
            _update_index(index_file, result.system_id, result.status)
            done += 1
            print(f"  [{done}/{len(pending)}] {result.system_id}: {result.status} "
                  f"({result.elapsed_s:.1f}s)")
    else:
        hb = _Heartbeat(len(pending), "systems").start()
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_evolve_one, w): w for w in work}
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    done += 1
                    hb.tick()
                    print(f"  [{done}/{len(pending)}] ERROR: {e}")
                    continue
                _save_jsonl(results_file, result)
                _update_index(index_file, result.system_id, result.status)
                done += 1
                hb.tick()
                print(f"  [{done}/{len(pending)}] {result.system_id}: {result.status} "
                      f"({result.elapsed_s:.1f}s)")
        hb.stop()

    elapsed = time.time() - t0
    print(f"Evolution complete: {done} systems in {elapsed:.1f}s")


def _calibrate_one(evo, n_years):
    """Worker function for parallel calibration (probe + spin)."""
    from chain_survey.perturbation_probe import run_probe
    from chain_survey.spin_survey import run_spin_survey
    probe = run_probe(
        evo.sim_archive_path, evo.hz_planet_idx, evo.system_id,
        n_years=n_years, thresholds=None,
    )
    spin = run_spin_survey(
        evo.sim_archive_path, evo.hz_planet_idx, evo.system_id,
        n_years=n_years,
    )
    return probe, spin


def _probe_one(args_tuple):
    """Worker function for parallel probing."""
    sim_path, hz_idx, sys_id, n_years, thresholds = args_tuple
    from chain_survey.perturbation_probe import run_probe
    return run_probe(sim_path, hz_idx, sys_id, n_years=n_years, thresholds=thresholds)


def cmd_calibrate(args):
    """Run calibration batch: probe + spin on first 400 systems."""
    evolutions = _load_jsonl(
        Path(args.output) / "evolution" / "results.jsonl", OrbitalResult,
    )
    survived = [r for r in evolutions if r.hz_planet_survived and r.sim_archive_path]
    n_cal = min(400, len(survived)) if not args.quick else min(5, len(survived))

    if n_cal == 0:
        print("No evolved systems with surviving HZ planets. Run --evolve first.")
        return

    cal_dir = Path(args.output) / "calibration"
    results_file = cal_dir / "spin_results.jsonl"
    print(f"Calibrating on {n_cal} systems...")

    t0 = time.time()
    n_years = 1000 if args.quick else 5000

    cal_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    batch = survived[:n_cal]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_calibrate_one, evo, n_years): evo for evo in batch}
        for future in as_completed(futures):
            evo = futures[future]
            try:
                probe, spin = future.result()
            except Exception as e:
                done += 1
                print(f"  [{done}/{n_cal}] {evo.system_id}: ERROR {e}")
                continue
            combined = {
                "system_id": evo.system_id,
                "probe": probe.to_dict(),
                "spin": spin.to_dict(),
            }
            with open(results_file, "a") as f:
                f.write(json.dumps(combined, default=_json_safe) + "\n")
            done += 1
            print(f"  [{done}/{n_cal}] {evo.system_id}: "
                  f"rms_dn={probe.rms_dn_over_n:.6f} flipflop={spin.is_flipflop}")

    elapsed = time.time() - t0
    print(f"Calibration complete: {n_cal} systems in {elapsed:.1f}s")


def cmd_learn_filter(args):
    """Learn perturbation filter thresholds from calibration data."""
    import numpy as np

    cal_file = Path(args.output) / "calibration" / "spin_results.jsonl"
    if not cal_file.exists():
        print("No calibration data. Run --calibrate first.")
        return

    records = []
    for line in cal_file.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))

    if not records:
        print("Empty calibration file.")
        return

    rms_vals = []
    flipflop_flags = []
    for rec in records:
        rms_vals.append(rec["probe"]["rms_dn_over_n"])
        flipflop_flags.append(rec["spin"]["is_flipflop"])

    rms_vals = np.array(rms_vals)
    flipflop_flags = np.array(flipflop_flags)

    n_ff = flipflop_flags.sum()
    print(f"Calibration: {len(records)} systems, {n_ff} flip-flops "
          f"({100*n_ff/len(records):.1f}%)")

    if n_ff == 0:
        print("WARNING: No flip-flops found in calibration. Filter will be disabled.")
        thresholds = None
    else:
        ff_rms = rms_vals[flipflop_flags]
        # Set thresholds at 5th/95th percentile of flip-flop systems
        # to capture 90% of flip-flops with some rejection
        rms_min = float(np.percentile(ff_rms, 5))
        rms_max = float(np.percentile(ff_rms, 95))

        # Expand range slightly for safety
        rms_min *= 0.5
        rms_max *= 2.0

        thresholds = {"min": rms_min, "max": rms_max}
        print(f"Learned thresholds: min={rms_min:.6f}, max={rms_max:.6f}")

        # Simple ROC AUC estimate
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(flipflop_flags, rms_vals)
            print(f"ROC AUC: {auc:.3f}")
            if auc < 0.6:
                print("WARNING: Poor predictive power. Filter will be disabled.")
                thresholds = None
        except Exception:
            print("Could not compute ROC AUC (sklearn not available or single class).")

    thresh_file = Path(args.output) / "calibration" / "thresholds.json"
    thresh_file.parent.mkdir(parents=True, exist_ok=True)
    thresh_file.write_text(json.dumps(thresholds))
    print(f"Saved: {thresh_file}")


def cmd_probe(args):
    """Run perturbation probes with filter."""
    evolutions = _load_jsonl(
        Path(args.output) / "evolution" / "results.jsonl", OrbitalResult,
    )
    survived = [r for r in evolutions if r.hz_planet_survived and r.sim_archive_path]

    # Load thresholds
    thresh_file = Path(args.output) / "calibration" / "thresholds.json"
    thresholds = None
    if thresh_file.exists():
        thresholds = json.loads(thresh_file.read_text())

    out_dir = Path(args.output) / "probes"
    results_file = out_dir / "results.jsonl"
    index_file = out_dir / "index.json"

    completed = _load_index(index_file) if args.resume else set()
    pending = [r for r in survived if r.system_id not in completed]
    n_years = 1000 if args.quick else 5000

    print(f"Probing {len(pending)}/{len(survived)} systems "
          f"(thresholds={'learned' if thresholds else 'none'})...")

    t0 = time.time()
    work = [(r.sim_archive_path, r.hz_planet_idx, r.system_id, n_years, thresholds)
            for r in pending]

    done = 0
    n_pass = 0

    if args.workers <= 1:
        for w in work:
            result = _probe_one(w)
            _save_jsonl(results_file, result)
            _update_index(index_file, result.system_id, result.filter_verdict)
            done += 1
            if result.filter_verdict == "PASS":
                n_pass += 1
            print(f"  [{done}/{len(pending)}] {result.system_id}: {result.filter_verdict}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_probe_one, w): w for w in work}
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    done += 1
                    continue
                _save_jsonl(results_file, result)
                _update_index(index_file, result.system_id, result.filter_verdict)
                done += 1
                if result.filter_verdict == "PASS":
                    n_pass += 1

    elapsed = time.time() - t0
    print(f"Probing complete: {done} systems, {n_pass} PASS in {elapsed:.1f}s")


def _spin_one(args_tuple):
    """Worker function for parallel spin survey."""
    sim_path, hz_idx, sys_id, mass, radius, mstar, q, triax, n_years = args_tuple
    from chain_survey.spin_survey import run_spin_survey
    return run_spin_survey(
        sim_path, hz_idx, sys_id,
        planet_mass_mearth=mass, planet_radius_rearth=radius,
        stellar_mass_msun=mstar, tidal_q=q, triaxiality=triax,
        n_years=n_years,
    )


def cmd_spin_survey(args):
    """Run Pipeline 2: spin dynamics survey."""
    probes = _load_jsonl(
        Path(args.output) / "probes" / "results.jsonl", PerturbationProbe,
    )
    passed = [p for p in probes if p.filter_verdict == "PASS"]

    # Load system data for planet properties
    systems = _load_jsonl(
        Path(args.output) / "systems" / "results.jsonl", SystemArchitecture,
    )
    sys_lookup = {s.system_id: s for s in systems}

    evolutions = _load_jsonl(
        Path(args.output) / "evolution" / "results.jsonl", OrbitalResult,
    )
    evo_lookup = {r.system_id: r for r in evolutions}

    out_dir = Path(args.output) / "spin_survey"
    results_file = out_dir / "results.jsonl"
    index_file = out_dir / "index.json"

    completed = _load_index(index_file) if args.resume else set()
    pending = [p for p in passed if p.system_id not in completed]
    n_years = 1000 if args.quick else 5000

    print(f"Spin survey: {len(pending)}/{len(passed)} systems...")

    t0 = time.time()
    work = []
    for p in pending:
        sys_arch = sys_lookup.get(p.system_id)
        evo = evo_lookup.get(p.system_id)
        if not sys_arch or not evo or not evo.sim_archive_path:
            continue
        hz_planet = sys_arch.planets[p.hz_planet_idx - 1]  # Convert 1-based to 0-based
        work.append((
            evo.sim_archive_path, p.hz_planet_idx, p.system_id,
            hz_planet.mass_mearth, hz_planet.radius_rearth,
            sys_arch.stellar_mass_msun, 22, 3e-5, n_years,
        ))

    done = 0
    n_ff = 0

    if args.workers <= 1:
        for w in work:
            result = _spin_one(w)
            _save_jsonl(results_file, result)
            _update_index(index_file, result.system_id, result.status)
            done += 1
            if result.is_flipflop:
                n_ff += 1
            print(f"  [{done}/{len(work)}] {result.system_id}: "
                  f"flipflop={result.is_flipflop} status={result.status}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_spin_one, w): w for w in work}
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    done += 1
                    continue
                _save_jsonl(results_file, result)
                _update_index(index_file, result.system_id, result.status)
                done += 1
                if result.is_flipflop:
                    n_ff += 1

    elapsed = time.time() - t0
    print(f"Spin survey complete: {done} systems, {n_ff} flip-flops in {elapsed:.1f}s")


def cmd_report(args):
    """Generate population statistics report."""
    base = Path(args.output)

    systems = _load_jsonl(base / "systems" / "results.jsonl", SystemArchitecture)
    evolutions = _load_jsonl(base / "evolution" / "results.jsonl", OrbitalResult)
    probes = _load_jsonl(base / "probes" / "results.jsonl", PerturbationProbe)
    spins = _load_jsonl(base / "spin_survey" / "results.jsonl", SpinSurveyResult)

    n_gen = len(systems)
    n_evo = len(evolutions)
    n_survived = sum(1 for r in evolutions if r.hz_planet_survived)
    n_probed = len(probes)
    n_pass = sum(1 for p in probes if p.filter_verdict == "PASS")
    n_spin = len(spins)
    n_ff = sum(1 for s in spins if s.is_flipflop)

    status_counts = {}
    for r in evolutions:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1

    report = f"""# Chain Survey Population Report

## Pipeline Summary

| Stage | Count |
|-------|-------|
| Generated | {n_gen} |
| Evolved | {n_evo} |
| HZ planet survived | {n_survived} |
| Probed | {n_probed} |
| Passed filter | {n_pass} |
| Spin surveyed | {n_spin} |
| **Flip-flop worlds** | **{n_ff}** |

## Evolution Outcomes

"""
    for status, count in sorted(status_counts.items()):
        report += f"- {status}: {count} ({100*count/n_evo:.1f}%)\n"

    if n_gen > 0 and n_ff > 0:
        rate = n_ff / n_gen
        report += f"\n## Flip-Flop Rate\n\n"
        report += f"- Raw: {n_ff}/{n_gen} = {100*rate:.2f}%\n"
        if n_survived > 0:
            report += f"- Among HZ survivors: {n_ff}/{n_survived} = {100*n_ff/n_survived:.2f}%\n"

    report_path = base / "population_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(report)
    print(f"\nSaved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Resonant chain survey orchestrator")
    parser.add_argument("--generate", type=int, default=0,
                        help="Generate N systems")
    parser.add_argument("--evolve", action="store_true",
                        help="Run Pipeline 1: orbital evolution")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run calibration batch")
    parser.add_argument("--learn-filter", action="store_true",
                        help="Learn filter thresholds from calibration")
    parser.add_argument("--probe", action="store_true",
                        help="Run perturbation probes")
    parser.add_argument("--spin-survey", action="store_true",
                        help="Run Pipeline 2: spin survey")
    parser.add_argument("--all", action="store_true",
                        help="Run full pipeline")
    parser.add_argument("--report", action="store_true",
                        help="Generate population report")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode (reduced samples)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers")
    parser.add_argument("--resume", action="store_true",
                        help="Skip completed systems")
    parser.add_argument("--output", type=str, default="results/chain_survey",
                        help="Output directory")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Per-system timeout (seconds)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base RNG seed")

    args = parser.parse_args()

    if args.output == "results/chain_survey":
        from datetime import datetime
        args.output = f"results/chain_survey_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if args.all:
        # Use explicit --generate if provided, else default
        if args.generate == 0:
            n = 50 if args.quick else 5000
            args.generate = n
        cmd_generate(args)
        cmd_evolve(args)
        cmd_calibrate(args)
        cmd_learn_filter(args)
        cmd_probe(args)
        cmd_spin_survey(args)
        cmd_report(args)
        return

    if args.generate > 0:
        cmd_generate(args)
    if args.evolve:
        cmd_evolve(args)
    if args.calibrate:
        cmd_calibrate(args)
    if args.learn_filter:
        cmd_learn_filter(args)
    if args.probe:
        cmd_probe(args)
    if args.spin_survey:
        cmd_spin_survey(args)
    if args.report:
        cmd_report(args)


if __name__ == "__main__":
    main()
