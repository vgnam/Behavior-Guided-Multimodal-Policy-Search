#!/usr/bin/env python3
"""Summarize the best reward from every air-hockey ProPS+ run."""

import argparse
import csv
import math
from pathlib import Path
import statistics
import sys


DEFAULT_PREFIX = "air_hockey_juggle_propsp"


def highest_total_reward(log_path: Path) -> float:
    """Return the largest finite value in an overall log's Total Reward column."""
    rewards = []
    with log_path.open("r", encoding="utf-8", newline="") as log_file:
        for row in csv.DictReader(log_file):
            normalized = {
                str(key).strip(): str(value).strip()
                for key, value in row.items()
                if key is not None and value is not None
            }
            raw_reward = normalized.get("Total Reward")
            if not raw_reward:
                continue
            try:
                reward = float(raw_reward)
            except ValueError:
                continue
            if math.isfinite(reward):
                rewards.append(reward)

    if not rewards:
        raise ValueError("no finite values found in the Total Reward column")
    return max(rewards)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Print the maximum Total Reward from each timestamped air-hockey "
            "ProPS+ overall_log.txt, followed by their average."
        )
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Directory containing timestamped run directories (default: logs)",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help=f"Run-directory prefix (default: {DEFAULT_PREFIX})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_paths = sorted(
        args.logs_dir.glob(f"{args.prefix}*/overall_log.txt")
    )
    if not log_paths:
        print(
            f"No overall logs matched "
            f"{args.logs_dir / (args.prefix + '*/overall_log.txt')}",
            file=sys.stderr,
        )
        return 1

    results = []
    for log_path in log_paths:
        try:
            best_reward = highest_total_reward(log_path)
        except (OSError, ValueError) as exc:
            print(f"Skipping {log_path}: {exc}", file=sys.stderr)
            continue
        results.append((log_path.parent.name, best_reward))

    if not results:
        print("No valid Total Reward values were found.", file=sys.stderr)
        return 1

    for run_name, best_reward in results:
        print(f"{run_name}: {best_reward:.12g}")

    average = statistics.fmean(reward for _, reward in results)
    print(f"\nAverage highest reward across {len(results)} runs: {average:.12g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
