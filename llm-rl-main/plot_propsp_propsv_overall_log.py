"""
Plot ProPS+, ProPS, and BMPS mean reward by iteration from overall_log.txt.

This script recursively searches for overall_log.txt under logs/, groups them
by problem and variant, and plots mean reward curves.

Usage:
    python plot_propsp_propsv_overall_log.py --problem acrobot
    python plot_propsp_propsv_overall_log.py --problem acrobot --save acrobot_compare.png
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUN_SUFFIX_RE = re.compile(r"(?:_seed_\d+|_run_\d+|_\d+)$", flags=re.IGNORECASE)
MODEL_SUFFIX_RE = re.compile(
    r"_(?:gpt|gemini|claude|mistral|llama|qwen|deepseek)[a-z0-9.\-]*$",
    flags=re.IGNORECASE,
)


def natural_key(text):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def strip_run_suffix(name):
    return RUN_SUFFIX_RE.sub("", name)


def strip_model_suffix(name):
    return MODEL_SUFFIX_RE.sub("", name)


def normalize_problem_name(folder_name):
    name = folder_name.lower().strip("_")
    name = strip_run_suffix(name)
    name = strip_model_suffix(name)
    name = re.sub(r"_propsp", "", name)
    name = re.sub(r"_propsv", "", name)
    name = re.sub(r"_props(?![a-z])", "", name)
    name = re.sub(r"_oneshot", "", name)
    name = re.sub(r"_rndm_proj", "", name)
    name = re.sub(
        r"mountain_?car_continuous",
        "mountaincarcontinuous",
        name,
        flags=re.IGNORECASE,
    )
    name = re.sub(
        r"inverted_double_pendulum",
        "inverteddoublependulum",
        name,
        flags=re.IGNORECASE,
    )
    return name.strip("_")


def classify_variant(folder_name):
    lower = folder_name.lower()
    if "_propsp" in lower:
        return "propsp"
    if "_propsv" in lower:
        return "propsv"
    if "_props" in lower and "_propsp" not in lower and "_propsv" not in lower:
        return "props"
    return None


def parse_overall_log_rewards(log_path):
    rewards = []
    try:
        with open(log_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.lower().startswith("iteration"):
                    continue
                parts = [part.strip() for part in line.split(",")]
                if len(parts) < 6:
                    continue
                try:
                    rewards.append(float(parts[5]))
                except ValueError:
                    continue
    except OSError:
        return None

    if not rewards:
        return None

    return np.asarray(rewards, dtype=float)


def discover_logs(logs_dir):
    """Recursively find all overall_log.txt files and classify them."""
    entries = []
    for log_path in logs_dir.rglob("overall_log.txt"):
        run_folder = log_path.parent.name
        variant = classify_variant(run_folder)
        if variant is None:
            continue
        problem = normalize_problem_name(run_folder)
        rewards = parse_overall_log_rewards(log_path)
        if rewards is None:
            continue
        entries.append({
            "problem": problem,
            "variant": variant,
            "run_name": run_folder,
            "rewards": rewards,
        })
    return entries


def aggregate_runs(runs):
    if not runs:
        return None
    min_len = min(len(r["rewards"]) for r in runs)
    matrix = np.stack([r["rewards"][:min_len] for r in runs], axis=0)
    return {
        "iters": np.arange(min_len),
        "mean": np.mean(matrix, axis=0),
        "std": np.std(matrix, axis=0),
        "n": len(runs),
        "min_len": min_len,
    }


def plot_problem(problem, data_dict, output_path=None):
    # Determine common length across all available variants
    min_len = min(d["min_len"] for d in data_dict.values())
    iters = np.arange(min_len)

    colors = {
        "props": "#2ca02c",   # green
        "propsp": "#1f77b4",  # blue
        "propsv": "#d62728",  # red
    }
    labels = {
        "props": "ProPS",
        "propsp": "ProPS+",
        "propsv": "BMPS",
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for variant in ["props", "propsp", "propsv"]:
        if variant not in data_dict:
            continue
        d = data_dict[variant]
        mean = d["mean"][:min_len]
        std = d["std"][:min_len]
        color = colors[variant]
        label = labels[variant]

        ax.plot(iters, mean, color=color, linewidth=2, label=label)
        ax.fill_between(
            iters,
            mean - std,
            mean + std,
            color=color,
            alpha=0.22,
        )

    # ax.set_title(f"{problem}", fontsize=14)  # no title per request
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Mean Reward", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend()
    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def discover_available_problems(entries):
    problems = set(e["problem"] for e in entries)
    return sorted(problems)


def main():
    parser = argparse.ArgumentParser(
        description="Compare ProPS+, ProPS, and BMPS mean reward curves from overall_log.txt."
    )
    parser.add_argument(
        "--problem",
        type=str,
        required=False,
        help="Normalized problem name, e.g. acrobot, lunarlander, mountaincarcontinuous.",
    )
    parser.add_argument(
        "--logs_dir",
        type=Path,
        default=Path(__file__).parent / "logs",
        help="Root logs directory (default: ./logs).",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save the plot to this path instead of displaying it.",
    )
    args = parser.parse_args()

    logs_dir = args.logs_dir
    if not logs_dir.exists():
        print(f"Logs directory not found: {logs_dir}")
        return

    entries = discover_logs(logs_dir)
    available_problems = discover_available_problems(entries)

    if not args.problem:
        print("Please provide --problem. Available problems:")
        for problem in available_problems:
            print(f"  - {problem}")
        return

    problem = args.problem.lower().strip()

    # Group by variant
    variant_runs = {"props": [], "propsp": [], "propsv": []}
    for e in entries:
        if e["problem"] != problem:
            continue
        if e["variant"] in variant_runs:
            variant_runs[e["variant"]].append(e)

    data_dict = {}
    for variant, runs in variant_runs.items():
        if runs:
            data_dict[variant] = aggregate_runs(runs)

    if not data_dict:
        print(f"Could not find any runs for problem '{problem}'.")
        print(f"Available problems: {available_problems}")
        return

    print(f"Problem: {problem}")
    for variant, data in data_dict.items():
        label = {"props": "ProPS", "propsp": "ProPS+", "propsv": "BMPS"}[variant]
        print(f"  {label}: {data['n']} runs, {data['min_len']} iterations")

    plot_problem(problem, data_dict, output_path=args.save)


if __name__ == "__main__":
    main()
