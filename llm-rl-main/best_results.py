"""
Show the best result for every log folder.

For each folder found under logs_dir:
  - Finds the episode with the highest total_reward in overall_log.txt
  - Reads rollout rewards from that episode's training_rollout.txt
  - Prints mean ± std, best episode index, and number of rollouts

Usage:
    python best_results.py
    python best_results.py --logs_dir logs
    python best_results.py --problems nav swimmer cartpole
    python best_results.py --sort reward          # sort by best mean reward (desc)
    python best_results.py --sort name            # sort by folder name (default)
"""

import os
import re
import argparse
import numpy as np


# ── Helpers ──────────────────────────────────────────────────────────────

def parse_overall_log(logdir):
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
    eps = []
    for name in os.listdir(logdir):
        m = re.match(r"episode_(\d+)$", name)
        if m:
            ep = int(m.group(1))
            if os.path.exists(os.path.join(logdir, f"episode_{ep}", "training_rollout.txt")):
                eps.append(ep)
    return sorted(eps)


def get_rollout_rewards(logdir, episode):
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


# ── Main ─────────────────────────────────────────────────────────────────

def main(logs_dir="logs", problems=None, sort_by="name"):
    if not os.path.isdir(logs_dir):
        print(f"[Error] Logs directory not found: {logs_dir}")
        return

    folders = sorted(
        f for f in os.listdir(logs_dir)
        if os.path.isdir(os.path.join(logs_dir, f))
    )

    # Filter by requested problems if specified
    if problems:
        folders = [
            f for f in folders
            if any(p.lower() in f.lower() for p in problems)
        ]

    if not folders:
        print("[Warning] No matching folders found.")
        return

    print(f"Scanning {len(folders)} folder(s) in '{logs_dir}'\n")
    print("=" * 70)

    results = []

    for folder in folders:
        path = os.path.join(logs_dir, folder)
        eps = completed_episode_indices(path)

        if not eps:
            results.append({
                "folder": folder,
                "status": "no_episodes",
                "best_ep": None,
                "mean": None,
                "std": None,
                "n_rollouts": 0,
                "n_eps": 0,
            })
            continue

        rows = parse_overall_log(path)

        # Build episode → total_reward map
        reward_map = {}
        for idx, ep in enumerate(eps):
            if idx < len(rows):
                reward_map[ep] = rows[idx]["total_reward"]

        if not reward_map:
            results.append({
                "folder": folder,
                "status": "no_reward_data",
                "best_ep": None,
                "mean": None,
                "std": None,
                "n_rollouts": 0,
                "n_eps": len(eps),
            })
            continue

        best_ep = max(reward_map, key=reward_map.get)
        best_log_reward = reward_map[best_ep]
        rollouts = get_rollout_rewards(path, best_ep)

        if rollouts:
            arr = np.array(rollouts)
            results.append({
                "folder": folder,
                "status": "ok",
                "best_ep": best_ep,
                "log_reward": best_log_reward,
                "mean": arr.mean(),
                "std": arr.std(),
                "n_rollouts": len(arr),
                "n_eps": len(eps),
            })
        else:
            results.append({
                "folder": folder,
                "status": "no_rollout",
                "best_ep": best_ep,
                "log_reward": best_log_reward,
                "mean": None,
                "std": None,
                "n_rollouts": 0,
                "n_eps": len(eps),
            })

    # Sort
    if sort_by == "reward":
        # Put folders with actual mean reward first (desc), then others
        results.sort(key=lambda r: r["mean"] if r["mean"] is not None else float("-inf"), reverse=True)
    # default: already sorted by folder name (alphabetical)

    # Print
    for r in results:
        folder = r["folder"]
        n_eps = r["n_eps"]

        if r["status"] == "no_episodes":
            print(f"  {folder}")
            print(f"    No completed episodes yet.")
        elif r["status"] == "no_reward_data":
            print(f"  {folder}  ({n_eps} eps)")
            print(f"    No reward data in overall_log.txt")
        elif r["status"] == "no_rollout":
            print(f"  {folder}  ({n_eps} eps)")
            print(f"    Best ep={r['best_ep']}  log_reward={r['log_reward']:.3f}  — no rollout data")
        else:
            print(f"  {folder}  ({n_eps} eps)")
            print(f"    Best ep={r['best_ep']:4d}  "
                  f"mean reward = {r['mean']:.3f} ± {r['std']:.3f}  "
                  f"(n={r['n_rollouts']} rollouts)")

        print()

    print("=" * 70)

    # Summary: best folders with actual results
    ok = [r for r in results if r["status"] == "ok"]
    if ok:
        top = max(ok, key=lambda r: r["mean"])
        print(f"\nOverall best: {top['folder']}  mean={top['mean']:.3f} ± {top['std']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Show best result for every log folder."
    )
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
        help="Only show folders matching these substrings (case-insensitive).",
    )
    parser.add_argument(
        "--sort",
        choices=["name", "reward"],
        default="name",
        help="Sort output by folder name (default) or by best mean reward.",
    )
    args = parser.parse_args()
    main(logs_dir=args.logs_dir, problems=args.problems, sort_by=args.sort)
