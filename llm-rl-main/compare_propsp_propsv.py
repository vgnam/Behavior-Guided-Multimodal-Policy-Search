"""
Compare ProPS+ (propsp) vs ProPS-V (propsv) variants for each matched problem.

For each pair found in the logs directory:
  - Finds the common iteration range (0 .. min_iters - 1)
  - Finds the best-performing iteration for each variant within that range
  - Reads all individual rollout rewards from that iteration's training_rollout.txt
  - Prints mean ± std of the best policy's rollout rewards

Usage:
    python compare_propsp_propsv.py [--logs_dir logs]
"""

import os
import re
import argparse
import numpy as np


# ── Helpers ──────────────────────────────────────────────────────────────

def parse_overall_log(logdir):
    """
    Return rows in order as list of dicts.
    Row index i corresponds to training episode i (0-based).
    """
    log_path = os.path.join(logdir, "overall_log.txt")
    if not os.path.exists(log_path):
        return []
    rows = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f.readlines()[1:]:          # skip header
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                try:
                    rows.append({
                        "iteration": int(float(parts[0])),
                        "total_reward": float(parts[5]),
                    })
                except ValueError:
                    continue
    return rows


def completed_episode_indices(logdir):
    """
    Return sorted list of episode numbers that have a training_rollout.txt.
    """
    eps = []
    for name in os.listdir(logdir):
        m = re.match(r"episode_(\d+)$", name)
        if m:
            ep = int(m.group(1))
            if os.path.exists(os.path.join(logdir, f"episode_{ep}", "training_rollout.txt")):
                eps.append(ep)
    return sorted(eps)


def get_rollout_rewards(logdir, episode):
    """
    Parse all 'Total reward: X' lines from episode_N/training_rollout.txt.
    Returns list of floats (one per evaluation rollout).
    """
    path = os.path.join(logdir, f"episode_{episode}", "training_rollout.txt")
    rewards = []
    if not os.path.exists(path):
        return rewards
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if "Total reward" in line:
                try:
                    rewards.append(float(line.split()[-1]))
                except ValueError:
                    pass
    return rewards


def normalize_name(folder_name):
    """
    Strip variant markers and minor naming differences to get a canonical key
    for matching propsp ↔ propsv.
    """
    name = folder_name
    # Remove variant markers (no \b: underscore is a word char, so \b won't fire before _)
    name = re.sub(r"_propsp", "", name)
    name = re.sub(r"_propsv", "", name)
    # Also handle folders named '..._props_...' or just ending in _props  (ProPS baseline)
    name = re.sub(r"_props(?![a-z])", "", name)
    # Normalise 'mountaincar_continuous' vs 'mountaincarcontinuous'
    name = re.sub(r"mountain_?car_continuous", "mountaincarcontinuous", name, flags=re.IGNORECASE)
    # Strip leading/trailing underscores left behind
    name = name.strip("_")
    return name


# ── Main ─────────────────────────────────────────────────────────────────

def main(logs_dir="logs", problems=None):
    if not os.path.isdir(logs_dir):
        print(f"[Error] Logs directory not found: {logs_dir}")
        return

    folders = sorted(
        f for f in os.listdir(logs_dir)
        if os.path.isdir(os.path.join(logs_dir, f))
    )

    # Classify folders
    propsp_map = {}   # normalized_key → (folder_name, full_path)
    propsv_map = {}

    for folder in folders:
        path = os.path.join(logs_dir, folder)
        if "_propsp" in folder:
            key = normalize_name(folder)
            propsp_map.setdefault(key, []).append((folder, path))
        elif "_propsv" in folder:
            key = normalize_name(folder)
            propsv_map.setdefault(key, []).append((folder, path))

    matched_keys = sorted(set(propsp_map) & set(propsv_map))

    # Filter by requested problems if specified
    if problems:
        problem_set = set(problems)
        matched_keys = [k for k in matched_keys
                        if any(p.lower() in k.lower() for p in problem_set)]

    if not matched_keys:
        print("[Warning] No matched propsp / propsv pairs found.")
        print(f"  propsp keys: {sorted(propsp_map)}")
        print(f"  propsv keys: {sorted(propsv_map)}")
        return

    print(f"Found {len(matched_keys)} matched pair(s)\n")
    print("=" * 70)

    for key in matched_keys:
        # If multiple candidates, pick the one with the most episodes
        def pick_best(candidates):
            best, best_path = candidates[0]
            best_count = len(completed_episode_indices(best_path))
            for fname, fpath in candidates[1:]:
                c = len(completed_episode_indices(fpath))
                if c > best_count:
                    best, best_path, best_count = fname, fpath, c
            return best, best_path

        propsp_folder, propsp_path = pick_best(propsp_map[key])
        propsv_folder, propsv_path = pick_best(propsv_map[key])

        propsp_eps = completed_episode_indices(propsp_path)
        propsv_eps = completed_episode_indices(propsv_path)

        if not propsp_eps or not propsv_eps:
            print(f"[{key}] Skipped – one or both variants have no completed episodes.")
            print("=" * 70)
            continue

        # Read overall_log rows (ordered = episode 0, 1, 2, ...)
        propsp_rows = parse_overall_log(propsp_path)
        propsv_rows = parse_overall_log(propsv_path)

        # Full range: each variant uses all its own logged episodes
        def build_reward_map_full(eps_list, rows):
            reward_map = {}
            for idx, ep in enumerate(eps_list):
                if idx < len(rows):
                    reward_map[ep] = rows[idx]["total_reward"]
            return reward_map

        propsp_rmap = build_reward_map_full(propsp_eps, propsp_rows)
        propsv_rmap = build_reward_map_full(propsv_eps, propsv_rows)

        # Find best episode over full range
        def find_best(rmap):
            if not rmap:
                return None, None
            best_ep = max(rmap, key=rmap.get)
            return best_ep, rmap[best_ep]

        propsp_best_ep, propsp_best_mean = find_best(propsp_rmap)
        propsv_best_ep, propsv_best_mean = find_best(propsv_rmap)

        # Get individual rollout rewards from the best episode
        propsp_rollouts = get_rollout_rewards(propsp_path, propsp_best_ep) if propsp_best_ep is not None else []
        propsv_rollouts = get_rollout_rewards(propsv_path, propsv_best_ep) if propsv_best_ep is not None else []

        # ── Print results ────────────────────────────────────────────────
        problem_label = key.replace("_gpt-oss", "").replace("_gemini-2.5-flash-lite", "")

        print(f"\nProblem : {problem_label}")
        print(f"  ProPS+ : {propsp_folder}  ({len(propsp_eps)} eps)")
        print(f"  ProPS-V: {propsv_folder}  ({len(propsv_eps)} eps)")
        print()

        if propsp_rollouts:
            arr = np.array(propsp_rollouts)
            print(f"  ProPS+  best ep={propsp_best_ep:4d}  "
                  f"mean reward = {arr.mean():.3f} ± {arr.std():.3f}  "
                  f"(n={len(arr)} rollouts)")
        else:
            print(f"  ProPS+  best ep={propsp_best_ep}  — no rollout data found")

        if propsv_rollouts:
            arr = np.array(propsv_rollouts)
            print(f"  ProPS-V best ep={propsv_best_ep:4d}  "
                  f"mean reward = {arr.mean():.3f} ± {arr.std():.3f}  "
                  f"(n={len(arr)} rollouts)")
        else:
            print(f"  ProPS-V best ep={propsv_best_ep}  — no rollout data found")

        # Summary comparison
        if propsp_rollouts and propsv_rollouts:
            pp_mean = np.mean(propsp_rollouts)
            pv_mean = np.mean(propsv_rollouts)
            winner = "ProPS-V" if pv_mean > pp_mean else "ProPS+"
            delta = abs(pv_mean - pp_mean)
            print(f"\n  → {winner} wins by {delta:.3f} (absolute mean reward difference)")

        print("=" * 70)

    # Report unmatched
    unmatched_propsp = sorted(set(propsp_map) - set(propsv_map))
    unmatched_propsv = sorted(set(propsv_map) - set(propsp_map))
    if unmatched_propsp:
        print(f"\nUnmatched ProPS+  : {[propsp_map[k][0][0] for k in unmatched_propsp]}")
    if unmatched_propsv:
        print(f"Unmatched ProPS-V : {[propsv_map[k][0][0] for k in unmatched_propsv]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="logs",
        help="Path to the logs directory (default: logs)",
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        default=None,
        help="Only show these problems (substring match). E.g. --problems nav swimmer",
    )
    args = parser.parse_args()
    main(logs_dir=args.logs_dir, problems=args.problems)
