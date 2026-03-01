"""
Plot mean ± std reward curves from experiment logs.

Two std sources (controlled by --std_source):
  rollouts  (default) — std across all rollouts within each iteration
                        (parsed from episode_N/training_rollout.txt)
  seeds               — std across multiple seed runs of the same experiment
                        (parsed from overall_log.txt; requires folder name convention
                         <exp>_seed_N or <exp>_N)

Usage:
    python plot_rewards.py                                  # scan logs/ folder
    python plot_rewards.py --logs_dir path/to/logs          # custom logs dir
    python plot_rewards.py --exps nav cartpole              # filter experiments
    python plot_rewards.py --std_source seeds               # use seed-level std
    python plot_rewards.py --smooth 5 --save out.png        # smoothing + save
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_seed_suffix(name: str) -> str:
    """Remove trailing _seed_N or _N suffix from experiment folder name."""
    name = re.sub(r"_seed_\d+$", "", name)
    name = re.sub(r"_\d+$", "", name)
    return name


def parse_overall_log(path: Path) -> np.ndarray | None:
    """
    Parse overall_log.txt → 1-D reward array indexed by iteration.

    Expected header:
        Iteration, CPU Time, API Time (LLM+VLM), Total Episodes, Total Steps, Total Reward
    """
    rewards: dict[int, float] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.lower().startswith("iteration"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 6:
                    continue
                try:
                    rewards[int(parts[0])] = float(parts[5])
                except ValueError:
                    continue
    except OSError:
        return None

    if not rewards:
        return None

    max_iter = max(rewards.keys())
    arr = np.full(max_iter + 1, np.nan)
    for it, r in rewards.items():
        arr[it] = r
    return arr


def parse_rollout_file(path: Path) -> list[float]:
    """Parse a training_rollout.txt and return all 'Total reward:' values."""
    rewards = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.lower().startswith("total reward:"):
                    try:
                        rewards.append(float(line.split(":", 1)[1].strip()))
                    except ValueError:
                        pass
    except OSError:
        pass
    return rewards


def parse_rollout_stats(exp_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Scan episode_N/ subdirectories for training_rollout.txt files.

    Returns (iterations, means, stds) arrays, or None if no data found.
    """
    pattern = re.compile(r"^episode_(\d+)$")
    data: dict[int, list[float]] = {}

    for sub in exp_dir.iterdir():
        m = pattern.match(sub.name)
        if not m:
            continue
        rollout_file = sub / "training_rollout.txt"
        if not rollout_file.exists():
            rollout_file = sub / "training_rollout_pre.txt"
        if not rollout_file.exists():
            continue
        rewards = parse_rollout_file(rollout_file)
        if rewards:
            data[int(m.group(1))] = rewards

    if not data:
        return None

    iters = np.array(sorted(data.keys()))
    means = np.array([np.mean(data[i]) for i in iters])
    stds = np.array([np.std(data[i]) for i in iters])
    return iters, means, stds


# ---------------------------------------------------------------------------
# Find experiments
# ---------------------------------------------------------------------------

def find_experiments_rollout(logs_dir: Path) -> dict[str, dict]:
    """
    Scan logs_dir for experiment dirs containing episode_N/ subdirs.

    Returns mapping: group_name -> {'iters': arr, 'means': arr, 'stds': arr}
    (one entry per group; multiple seeds are averaged together naively by
    aligning on iteration index).
    """
    groups: dict[str, list[tuple]] = defaultdict(list)

    for exp_dir in sorted(logs_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        result = parse_rollout_stats(exp_dir)
        if result is None:
            continue
        group = strip_seed_suffix(exp_dir.name)
        groups[group].append(result)
        print(f"  Loaded  {exp_dir.name}  ({len(result[0])} iters, rollout-level std)")

    # Merge seeds by interpolating onto common iter grid
    merged: dict[str, dict] = {}
    for group, runs in groups.items():
        if len(runs) == 1:
            iters, means, stds = runs[0]
            merged[group] = {"iters": iters, "means": means, "stds": stds, "n": 1}
        else:
            min_iters = min(r[0][-1] for r in runs)
            common = np.arange(0, min_iters + 1)
            all_means = []
            for iters, means, stds in runs:
                interp = np.interp(common, iters, means)
                all_means.append(interp)
            mat = np.stack(all_means)
            merged[group] = {
                "iters": common,
                "means": np.mean(mat, axis=0),
                "stds": np.std(mat, axis=0),
                "n": len(runs),
            }
    return merged


def find_experiments_seeds(logs_dir: Path) -> dict[str, dict]:
    """
    Scan logs_dir for overall_log.txt files and group by seed suffix.

    Returns mapping: group_name -> {'iters', 'means', 'stds', 'n'}
    """
    groups: dict[str, list[np.ndarray]] = defaultdict(list)

    for log_file in sorted(logs_dir.glob("*/overall_log.txt")):
        exp_name = log_file.parent.name
        group = strip_seed_suffix(exp_name)
        arr = parse_overall_log(log_file)
        if arr is not None:
            groups[group].append(arr)
            print(f"  Loaded  {log_file}  ({len(arr)} iters, seed-level std)")
        else:
            print(f"  Skipped {log_file}  (parse failed)")

    merged: dict[str, dict] = {}
    for group, arrays in groups.items():
        min_len = min(len(a) for a in arrays)
        matrix = np.stack([a[:min_len] for a in arrays], axis=0)
        merged[group] = {
            "iters": np.arange(min_len),
            "means": np.nanmean(matrix, axis=0),
            "stds": np.nanstd(matrix, axis=0),
            "n": len(arrays),
        }
    return merged


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def smooth_arrays(
    iters: np.ndarray, means: np.ndarray, stds: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if window <= 1:
        return iters, means, stds
    kernel = np.ones(window) / window
    means = np.convolve(means, kernel, mode="valid")
    stds = np.convolve(stds, kernel, mode="valid")
    iters = iters[window - 1:]
    return iters, means, stds


def plot_groups(
    groups: dict[str, dict],
    filter_names: list[str] | None = None,
    smooth_window: int = 1,
    output_path: Path | None = None,
    std_source: str = "rollouts",
) -> None:
    if filter_names:
        groups = {
            k: v for k, v in groups.items()
            if any(f.lower() in k.lower() for f in filter_names)
        }

    if not groups:
        print("No experiments to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab10
    colors = [cmap(i % 10) for i in range(len(groups))]

    for color, (group_name, data) in zip(colors, sorted(groups.items())):
        iters, means, stds = data["iters"], data["means"], data["stds"]
        n = data["n"]
        iters, means, stds = smooth_arrays(iters, means, stds, smooth_window)

        src_label = f"rollout std" if std_source == "rollouts" else f"n={n} seeds"
        label = f"{group_name} ({src_label})"

        ax.plot(iters, means, color=color, linewidth=2, label=label)
        ax.fill_between(iters, means - stds, means + stds, color=color, alpha=0.2)

    ax.set_xlabel("Iteration", fontsize=13)
    ax.set_ylabel("Total Reward", fontsize=13)
    title_src = "Rollout variance" if std_source == "rollouts" else "Seed variance"
    ax.set_title(f"Reward Curves — Mean ± Std  ({title_src})", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot reward curves from logs.")
    parser.add_argument(
        "--logs_dir", type=Path, default=Path(__file__).parent / "logs",
        help="Root logs directory (default: ./logs)",
    )
    parser.add_argument(
        "--exps", nargs="*", default=None,
        help="Filter experiments whose name contains these substrings.",
    )
    parser.add_argument(
        "--smooth", type=int, default=1,
        help="Rolling-average window for smoothing (default: 1 = off).",
    )
    parser.add_argument(
        "--save", type=Path, default=None,
        help="Save figure to this path instead of displaying it.",
    )
    parser.add_argument(
        "--std_source", choices=["rollouts", "seeds"], default="rollouts",
        help="Source of variance: 'rollouts' (within-iter) or 'seeds' (cross-run). Default: rollouts.",
    )
    args = parser.parse_args()

    logs_dir: Path = args.logs_dir
    if not logs_dir.exists():
        print(f"Logs directory not found: {logs_dir}")
        return

    print(f"Scanning {logs_dir} (std_source={args.std_source}) ...")

    if args.std_source == "rollouts":
        groups = find_experiments_rollout(logs_dir)
    else:
        groups = find_experiments_seeds(logs_dir)

    if not groups:
        print("No experiment data found.")
        return

    print(f"\nFound {len(groups)} experiment group(s): {list(groups.keys())}")
    plot_groups(
        groups,
        filter_names=args.exps,
        smooth_window=args.smooth,
        output_path=args.save,
        std_source=args.std_source,
    )


if __name__ == "__main__":
    main()


# C:/Users/user/AppData/Local/Programs/Python/Python311/python.exe plot_rewards.py --exps mountaincar

# cd d:\Prompted-Policy-Search\llm-rl-main; C:/Users/user/AppData/Local/Programs/Python/Python311/python.exe plot_rewards.py --exps nav