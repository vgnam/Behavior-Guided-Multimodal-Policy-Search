"""
Plot ProPS+ vs ProPS-V mean reward by iteration from overall_log.txt.

This script aggregates up to N runs for each variant, computes the mean reward
curve across runs, and draws a variance shade (mean +/- std).

Important:
- It uses row order in overall_log.txt as the iteration axis, not the logged
  iteration number. This avoids off-by-one mismatches between runners that log
  iterations starting from 0 and runners that log starting from 1.
- It only reads overall_log.txt, as requested.

Examples:
    python plot_propsp_propsv_overall_log.py --problem acrobot
    python plot_propsp_propsv_overall_log.py --problem acrobot --max_runs 10 --save acrobot_compare.png
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
    if "_propsv" in lower and "_oneshot" not in lower and "_rndm_proj" not in lower:
        return "propsv"
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


def collect_runs(logs_dir, problem, variant, max_runs):
    candidates = []
    normalized_problem = problem.lower().strip()

    for folder in sorted((path for path in logs_dir.iterdir() if path.is_dir()), key=lambda p: natural_key(p.name)):
        folder_variant = classify_variant(folder.name)
        if folder_variant != variant:
            continue

        if normalize_problem_name(folder.name) != normalized_problem:
            continue

        overall_log = folder / "overall_log.txt"
        rewards = parse_overall_log_rewards(overall_log)
        if rewards is None:
            continue

        candidates.append((folder.name, rewards))

    if len(candidates) > max_runs:
        candidates = candidates[:max_runs]

    return candidates


def aggregate_runs(runs):
    if not runs:
        return None

    min_len = min(len(rewards) for _, rewards in runs)
    matrix = np.stack([rewards[:min_len] for _, rewards in runs], axis=0)
    iters = np.arange(min_len)

    return {
        "iters": iters,
        "mean": np.mean(matrix, axis=0),
        "std": np.std(matrix, axis=0),
        "n": len(runs),
        "run_names": [name for name, _ in runs],
        "min_len": min_len,
    }


def discover_available_problems(logs_dir):
    problems = set()
    for folder in logs_dir.iterdir():
        if not folder.is_dir():
            continue
        if classify_variant(folder.name) is None:
            continue
        problems.add(normalize_problem_name(folder.name))
    return sorted(problems)


def plot_problem(problem, propsp_data, propsv_data, output_path=None):
    common_len = min(propsp_data["min_len"], propsv_data["min_len"])
    iters = np.arange(common_len)

    propsp_mean = propsp_data["mean"][:common_len]
    propsp_std = propsp_data["std"][:common_len]
    propsv_mean = propsv_data["mean"][:common_len]
    propsv_std = propsv_data["std"][:common_len]

    fig, ax = plt.subplots(figsize=(10, 6))

    propsp_color = "#1f77b4"
    propsv_color = "#d62728"

    ax.plot(iters, propsp_mean, color=propsp_color, linewidth=2, label=f"ProPS+ (n={propsp_data['n']})")
    ax.fill_between(
        iters,
        propsp_mean - propsp_std,
        propsp_mean + propsp_std,
        color=propsp_color,
        alpha=0.22,
    )

    ax.plot(iters, propsv_mean, color=propsv_color, linewidth=2, label=f"ProPS-V (n={propsv_data['n']})")
    ax.fill_between(
        iters,
        propsv_mean - propsv_std,
        propsv_mean + propsv_std,
        color=propsv_color,
        alpha=0.22,
    )

    ax.set_title(f"{problem}: mean reward by iteration from overall_log.txt", fontsize=14)
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


def main():
    parser = argparse.ArgumentParser(
        description="Compare ProPS+ vs ProPS-V mean reward curves from overall_log.txt with variance shading."
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
        "--max_runs",
        type=int,
        default=10,
        help="Maximum runs to use per variant (default: 10).",
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

    available_problems = discover_available_problems(logs_dir)
    if not args.problem:
        print("Please provide --problem. Available problems:")
        for problem in available_problems:
            print(f"  - {problem}")
        return

    problem = args.problem.lower().strip()
    propsp_runs = collect_runs(logs_dir, problem, "propsp", args.max_runs)
    propsv_runs = collect_runs(logs_dir, problem, "propsv", args.max_runs)

    if not propsp_runs or not propsv_runs:
        print(f"Could not find both ProPS+ and ProPS-V runs for problem '{problem}'.")
        print(f"Available problems: {available_problems}")
        return

    propsp_data = aggregate_runs(propsp_runs)
    propsv_data = aggregate_runs(propsv_runs)

    print(f"Problem: {problem}")
    print(f"  ProPS+ runs used ({propsp_data['n']}): {propsp_data['run_names']}")
    print(f"  ProPS-V runs used ({propsv_data['n']}): {propsv_data['run_names']}")
    print(f"  Common plotted iterations: {min(propsp_data['min_len'], propsv_data['min_len'])}")

    plot_problem(problem, propsp_data, propsv_data, output_path=args.save)


if __name__ == "__main__":
    main()
