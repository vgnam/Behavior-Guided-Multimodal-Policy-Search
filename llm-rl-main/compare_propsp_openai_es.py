"""
Compare ProPS+ (propsp) vs OpenAI-ES on a fair shared budget.

By default, fairness is defined by cumulative environment interaction budget:
the script compares the best policy each method found while staying within the
same `Total Steps` budget. This avoids giving OpenAI-ES extra credit simply
because each outer iteration evaluates many more candidates.

Usage:
    python compare_propsp_openai_es.py
    python compare_propsp_openai_es.py --problems acrobot nav
    python compare_propsp_openai_es.py --budget_key episodes
    python compare_propsp_openai_es.py --propsp logs/acrobot_propsp_gpt-oss --openai_es logs/acrobot_openai_es
"""

import argparse
import os
import re

import numpy as np


MODEL_SUFFIX_RE = re.compile(
    r"_(gpt|gemini|claude|mistral|llama|qwen|deepseek)[a-z0-9.\-]*$",
    flags=re.IGNORECASE,
)

BUDGET_FIELDS = {
    "steps": "total_steps",
    "episodes": "total_episodes",
    "iterations": "iteration_index",
}


def parse_overall_log(logdir):
    log_path = os.path.join(logdir, "overall_log.txt")
    if not os.path.exists(log_path):
        return []

    rows = []
    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle.readlines()[1:]:
            line = line.strip()
            if not line:
                continue

            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 6:
                continue

            try:
                rows.append(
                    {
                        "logged_iteration": int(float(parts[0])),
                        "cpu_time": float(parts[1]),
                        "api_time": float(parts[2]),
                        "total_episodes": int(float(parts[3])),
                        "total_steps": int(float(parts[4])),
                        "total_reward": float(parts[5]),
                    }
                )
            except ValueError:
                continue

    return rows


def completed_episode_indices(logdir):
    episodes = []
    if not os.path.isdir(logdir):
        return episodes

    for name in os.listdir(logdir):
        match = re.match(r"episode_(\d+)$", name)
        if not match:
            continue

        episode = int(match.group(1))
        rollout_path = os.path.join(logdir, name, "training_rollout.txt")
        if os.path.exists(rollout_path):
            episodes.append(episode)

    return sorted(episodes)


def build_progress_entries(logdir):
    rows = parse_overall_log(logdir)
    episodes = completed_episode_indices(logdir)
    aligned_count = min(len(rows), len(episodes))

    entries = []
    for idx in range(aligned_count):
        row = rows[idx]
        entries.append(
            {
                "episode": episodes[idx],
                "iteration_index": idx + 1,
                "total_episodes": row["total_episodes"],
                "total_steps": row["total_steps"],
                "total_reward": row["total_reward"],
            }
        )

    return entries


def get_rollout_rewards(logdir, episode):
    rewards = []
    rollout_path = os.path.join(logdir, f"episode_{episode}", "training_rollout.txt")
    if not os.path.exists(rollout_path):
        return rewards

    with open(rollout_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if "Total reward" not in line:
                continue
            try:
                rewards.append(float(line.split()[-1]))
            except ValueError:
                continue

    return rewards


def summarize_rewards(rewards):
    arr = np.asarray(rewards, dtype=float)
    return float(arr.mean()), float(arr.std())


def normalize_name(name):
    normalized = name.lower().strip("_")
    normalized = normalized.replace("_propsp", "")
    normalized = normalized.replace("_openai_es", "")
    normalized = re.sub(
        r"mountain_?car_continuous",
        "mountaincarcontinuous",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"inverted_double_pendulum",
        "inverteddoublependulum",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = MODEL_SUFFIX_RE.sub("", normalized)
    return normalized.strip("_")


def classify_logdirs(logs_dir):
    propsp_map = {}
    openai_es_map = {}

    if not os.path.isdir(logs_dir):
        return propsp_map, openai_es_map

    for folder in sorted(os.listdir(logs_dir)):
        path = os.path.join(logs_dir, folder)
        if not os.path.isdir(path):
            continue

        folder_lower = folder.lower()
        if "_propsp" in folder_lower:
            key = normalize_name(folder)
            propsp_map.setdefault(key, []).append((folder, path))
        elif "_openai_es" in folder_lower:
            key = normalize_name(folder)
            openai_es_map.setdefault(key, []).append((folder, path))

    return propsp_map, openai_es_map


def pick_best_candidate(candidates):
    def sort_key(candidate):
        _, path = candidate
        entries = build_progress_entries(path)
        if not entries:
            return (0, 0, float("-inf"))
        final_entry = entries[-1]
        return (
            len(entries),
            final_entry["total_steps"],
            final_entry["total_reward"],
        )

    return max(candidates, key=sort_key)


def best_entry_under_budget(entries, budget_field, budget_limit):
    eligible = [entry for entry in entries if entry[budget_field] <= budget_limit]
    if not eligible:
        return None

    return max(
        eligible,
        key=lambda entry: (
            entry["total_reward"],
            -entry[budget_field],
            -entry["episode"],
        ),
    )


def compare_pair(problem_label, propsp_pair, openai_es_pair, budget_key):
    budget_field = BUDGET_FIELDS[budget_key]

    propsp_folder, propsp_path = propsp_pair
    es_folder, es_path = openai_es_pair

    propsp_entries = build_progress_entries(propsp_path)
    es_entries = build_progress_entries(es_path)

    if not propsp_entries or not es_entries:
        print(f"\nProblem : {problem_label}")
        if not propsp_entries:
            print(f"  ProPS+    : {propsp_folder} - no aligned reward data")
        if not es_entries:
            print(f"  OpenAI-ES : {es_folder} - no aligned reward data")
        print("=" * 70)
        return

    common_budget = min(propsp_entries[-1][budget_field], es_entries[-1][budget_field])
    propsp_best = best_entry_under_budget(propsp_entries, budget_field, common_budget)
    es_best = best_entry_under_budget(es_entries, budget_field, common_budget)

    print(f"\nProblem : {problem_label}")
    print(f"  Fair budget ({budget_field}) <= {common_budget}")
    print(f"  ProPS+    : {propsp_folder}")
    print(f"  OpenAI-ES : {es_folder}")
    print()

    method_rows = []
    for label, logdir, entry in [
        ("ProPS+", propsp_path, propsp_best),
        ("OpenAI-ES", es_path, es_best),
    ]:
        if entry is None:
            print(f"  {label:<9} no episode fits the shared budget")
            method_rows.append((label, None))
            continue

        rollout_rewards = get_rollout_rewards(logdir, entry["episode"])
        if rollout_rewards:
            mean_reward, std_reward = summarize_rewards(rollout_rewards)
            reward_text = (
                f"mean reward = {mean_reward:.3f} +/- {std_reward:.3f}"
                f"  (n={len(rollout_rewards)} rollouts)"
            )
            compare_score = mean_reward
        else:
            reward_text = f"logged mean reward = {entry['total_reward']:.3f}"
            compare_score = entry["total_reward"]

        print(
            f"  {label:<9} best ep={entry['episode']:4d}  "
            f"{budget_field}={entry[budget_field]:7d}  {reward_text}"
        )
        method_rows.append((label, compare_score))

    valid_rows = [(label, score) for label, score in method_rows if score is not None]
    if len(valid_rows) == 2:
        valid_rows.sort(key=lambda item: item[1], reverse=True)
        winner, best_score = valid_rows[0]
        _, second_score = valid_rows[1]
        print(
            f"\n  -> {winner} wins by {abs(best_score - second_score):.3f} "
            f"(absolute mean reward difference under the shared {budget_field} budget)"
        )

    print("=" * 70)


def compare_explicit_pair(propsp_logdir, openai_es_logdir, budget_key):
    propsp_folder = os.path.basename(os.path.normpath(propsp_logdir))
    es_folder = os.path.basename(os.path.normpath(openai_es_logdir))

    propsp_pair = (propsp_folder, propsp_logdir)
    openai_es_pair = (es_folder, openai_es_logdir)
    problem_label = normalize_name(propsp_folder) or normalize_name(es_folder)
    compare_pair(problem_label, propsp_pair, openai_es_pair, budget_key)


def compare_all_pairs(logs_dir, problems, budget_key):
    propsp_map, openai_es_map = classify_logdirs(logs_dir)
    matched_keys = sorted(set(propsp_map) & set(openai_es_map))

    if problems:
        matched_keys = [
            key
            for key in matched_keys
            if any(problem.lower() in key for problem in problems)
        ]

    if not matched_keys:
        print("[Warning] No matched ProPS+ / OpenAI-ES pairs found.")
        print(f"  ProPS+ keys    : {sorted(propsp_map)}")
        print(f"  OpenAI-ES keys : {sorted(openai_es_map)}")
        return

    print(f"Found {len(matched_keys)} matched ProPS+ / OpenAI-ES pair(s)")
    print("=" * 70)

    for key in matched_keys:
        propsp_pair = pick_best_candidate(propsp_map[key])
        openai_es_pair = pick_best_candidate(openai_es_map[key])
        compare_pair(key, propsp_pair, openai_es_pair, budget_key)

    unmatched_propsp = sorted(set(propsp_map) - set(openai_es_map))
    unmatched_openai_es = sorted(set(openai_es_map) - set(propsp_map))
    if unmatched_propsp:
        print(f"\nUnmatched ProPS+    : {unmatched_propsp}")
    if unmatched_openai_es:
        print(f"Unmatched OpenAI-ES : {unmatched_openai_es}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="logs",
        help="Path to the logs directory for auto-discovery mode.",
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        default=None,
        help="Only compare problems whose normalized name contains one of these substrings.",
    )
    parser.add_argument(
        "--budget_key",
        choices=sorted(BUDGET_FIELDS),
        default="steps",
        help="Shared budget used for the fair comparison.",
    )
    parser.add_argument(
        "--propsp",
        type=str,
        default=None,
        help="Explicit ProPS+ logdir. If set, --openai_es must also be set.",
    )
    parser.add_argument(
        "--openai_es",
        type=str,
        default=None,
        help="Explicit OpenAI-ES logdir. If set, --propsp must also be set.",
    )
    args = parser.parse_args()

    if bool(args.propsp) != bool(args.openai_es):
        raise ValueError("--propsp and --openai_es must be provided together")

    if args.propsp and args.openai_es:
        compare_explicit_pair(args.propsp, args.openai_es, args.budget_key)
    else:
        compare_all_pairs(args.logs_dir, args.problems, args.budget_key)


if __name__ == "__main__":
    main()
