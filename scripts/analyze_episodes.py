#!/usr/bin/env python3
"""Analyze episode-level temporal patterns in PTB worlds.

Usage:
    uv run python scripts/analyze_episodes.py [--input results/sensitivity_analysis_v33]

Output:
    Terminal summary of world categories and top "slow bouncer" candidates.

Requirements:
    - results.jsonl files must have episode data (re-run with episode extraction)

Categories:
    - persistent_lock: One TL state > 90%, no PTB
    - slow_bouncer_fast_ptb: TL > 80%, PTB transitions < 10yr
    - slow_bouncer_slow_ptb: TL > 80%, PTB transitions 10-20yr
    - rapid_flipper: Multiple TL episodes (> 5 flips)
    - ptb_dominant: PTB > 50%
    - mixed: No clear pattern
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_results_with_episodes(jsonl_path: Path) -> list[dict]:
    """Load only results that have episode data."""
    results = []
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            if data.get("status") == "OK" and "episodes" in data and data["episodes"]:
                results.append(data)
    return results


def find_bounce_patterns(episodes: list[dict], ptb_max_yr: float = 20) -> list[dict]:
    """Find TL0 ↔ TLπ flip-flop patterns with short PTB transitions.

    Pattern: TL0 → (short PTB) → TLπ → (short PTB) → TL0

    Args:
        episodes: List of episode dicts with {type, duration_yr, ...}
        ptb_max_yr: Maximum PTB duration to consider "short"

    Returns:
        List of bounce pattern dicts with episode details.
    """
    patterns = []
    i = 0
    while i < len(episodes) - 2:
        ep = episodes[i]

        # Look for TL0 → PTB → TLπ or TLπ → PTB → TL0
        if ep["type"] in ("TL_ZERO", "TL_PI"):
            next_ep = episodes[i + 1]
            after_ep = episodes[i + 2] if i + 2 < len(episodes) else None

            if (next_ep["type"] == "PTB" and
                next_ep["duration_yr"] <= ptb_max_yr and
                after_ep is not None and
                after_ep["type"] in ("TL_ZERO", "TL_PI") and
                after_ep["type"] != ep["type"]):

                patterns.append({
                    "start_type": ep["type"],
                    "start_duration_yr": ep["duration_yr"],
                    "ptb_duration_yr": next_ep["duration_yr"],
                    "end_type": after_ep["type"],
                    "end_duration_yr": after_ep["duration_yr"],
                    "t_start_yr": ep["t_start_yr"],
                })
                i += 3  # Skip past the pattern
                continue

        i += 1

    return patterns


def classify_temporal_behavior(result: dict) -> str:
    """Classify world by episode temporal pattern."""
    episodes = result.get("episodes", [])
    if not episodes:
        return "unknown"

    fractions = result["fractions"]
    tl_total = fractions.get("TL_ZERO", 0) + fractions.get("TL_PI", 0)
    ptb_frac = fractions.get("PTB", 0)

    # Count episodes by type
    tl0_count = sum(1 for e in episodes if e["type"] == "TL_ZERO")
    tlpi_count = sum(1 for e in episodes if e["type"] == "TL_PI")
    ptb_count = sum(1 for e in episodes if e["type"] == "PTB")

    # Classify
    if tl_total > 0.9 and ptb_count == 0:
        return "persistent_lock"

    if tl_total > 0.8 and ptb_count > 0:
        patterns = find_bounce_patterns(episodes, ptb_max_yr=20)
        if patterns:
            avg_ptb = np.mean([p["ptb_duration_yr"] for p in patterns])
            if avg_ptb < 10:
                return "slow_bouncer_fast_ptb"  # <10yr PTB
            else:
                return "slow_bouncer_slow_ptb"  # 10-20yr PTB
        return "tl_with_sporadic_ptb"

    if (tl0_count + tlpi_count) > 5:
        return "rapid_flipper"

    if ptb_frac > 0.5:
        return "ptb_dominant"

    return "mixed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/sensitivity_analysis_v33/",
                        help="Input directory with study results")
    args = parser.parse_args()

    input_dir = Path(args.input)

    # Load results from all studies
    all_results = []
    for study in ["study1", "study2", "study3", "study4"]:
        study_file = input_dir / study / "results.jsonl"
        if not study_file.exists():
            continue
        results = load_results_with_episodes(study_file)
        all_results.extend(results)
        print(f"Loaded {len(results)} results with episodes from {study}")

    if not all_results:
        print("\nNo results with episode data found. Re-run configs with episode extraction:")
        print("  uv run python scripts/sensitivity_analysis.py --resume-configs <config_ids.txt>")
        sys.exit(1)

    print(f"\nTotal: {len(all_results)} results with episode data\n")

    # Categorize
    categories = defaultdict(list)
    for result in all_results:
        cat = classify_temporal_behavior(result)
        categories[cat].append(result)

    print("=== Temporal Behavior Categories ===")
    for cat, worlds in sorted(categories.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"{cat:25s}: {len(worlds):3d} worlds")

    # Focus on slow bouncers with fast PTB
    print("\n=== Slow Bouncers with Fast PTB (<10 yr transitions) ===")
    slow_bouncers = categories.get("slow_bouncer_fast_ptb", [])

    if not slow_bouncers:
        print("No slow bouncer worlds found")
        return

    for i, result in enumerate(sorted(slow_bouncers, key=lambda r: r["quality"]["overall_score"], reverse=True)[:20]):
        cfg = result["config"]
        episodes = result["episodes"]
        patterns = find_bounce_patterns(episodes, ptb_max_yr=10)

        if patterns:
            avg_tl = np.mean([p["start_duration_yr"] for p in patterns] + [p["end_duration_yr"] for p in patterns])
            avg_ptb = np.mean([p["ptb_duration_yr"] for p in patterns])

            print(f"\n{i+1}. Q={cfg['q']} a={cfg['distance_au']:.4f} AU triax={cfg['triax']:.2e}")
            print(f"   {len(patterns)} flip-flop(s): TL~{avg_tl:.1f}yr, PTB~{avg_ptb:.1f}yr")
            print(f"   T_term={result['temps']['t_term']}K, score={result['quality']['overall_score']:.3f}")

            # Show first pattern as example
            if patterns:
                p = patterns[0]
                print(f"   Example: {p['start_type']} ({p['start_duration_yr']:.1f}yr) → "
                      f"PTB ({p['ptb_duration_yr']:.1f}yr) → "
                      f"{p['end_type']} ({p['end_duration_yr']:.1f}yr)")


if __name__ == "__main__":
    main()
