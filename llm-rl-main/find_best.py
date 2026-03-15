"""
Find the best Total Reward in overall_log.txt for one or more log folders.

Usage:
    python find_best.py                          # scan all folders under logs/
    python find_best.py --logs_dir logs
    python find_best.py --folder logs/cartpole_propsp_gpt-oss
    python find_best.py --problems cartpole ant  # filter by name substring
    python find_best.py --sort reward            # sort output by best reward (desc)
"""

import os
import argparse


def parse_overall_log(path):
    """Read overall_log.txt and return list of (iteration, total_reward) tuples."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines()[1:]:          # skip header
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                try:
                    rows.append((int(float(parts[0])), float(parts[5])))
                except ValueError:
                    continue
    return rows


def best_in_folder(folder_path):
    """Return (best_iteration, best_reward, total_rows) for a log folder."""
    log_path = os.path.join(folder_path, "overall_log.txt")
    rows = parse_overall_log(log_path)
    if not rows:
        return None, None, 0
    best_iter, best_reward = max(rows, key=lambda r: r[1])
    return best_iter, best_reward, len(rows)


def main():
    parser = argparse.ArgumentParser(description="Find best reward in overall_log.txt")
    parser.add_argument("--logs_dir", default="logs",
                        help="Root directory containing log folders (default: logs)")
    parser.add_argument("--folder", default=None,
                        help="Path to a single log folder (overrides --logs_dir)")
    parser.add_argument("--problems", nargs="*", default=None,
                        help="Filter folders by substring (e.g. cartpole ant)")
    parser.add_argument("--sort", choices=["name", "reward"], default="name",
                        help="Sort output by name or reward (desc)")
    args = parser.parse_args()

    # ── Single folder mode ────────────────────────────────────────────────
    if args.folder:
        folder_path = args.folder.rstrip("/\\")
        best_iter, best_reward, n_rows = best_in_folder(folder_path)
        folder_name = os.path.basename(folder_path)
        if best_reward is None:
            print(f"[{folder_name}]  no overall_log.txt or no data found")
        else:
            print(f"[{folder_name}]")
            print(f"  Best reward   : {best_reward}")
            print(f"  At iteration  : {best_iter}")
            print(f"  Total rows    : {n_rows}")
        return

    # ── Multi-folder mode ─────────────────────────────────────────────────
    if not os.path.isdir(args.logs_dir):
        print(f"[Error] Directory not found: {args.logs_dir}")
        return

    folders = sorted(
        f for f in os.listdir(args.logs_dir)
        if os.path.isdir(os.path.join(args.logs_dir, f))
    )

    if args.problems:
        folders = [
            f for f in folders
            if any(p.lower() in f.lower() for p in args.problems)
        ]

    if not folders:
        print("[Warning] No matching folders found.")
        return

    results = []
    for folder in folders:
        folder_path = os.path.join(args.logs_dir, folder)
        best_iter, best_reward, n_rows = best_in_folder(folder_path)
        results.append({
            "folder":      folder,
            "best_reward": best_reward,
            "best_iter":   best_iter,
            "n_rows":      n_rows,
        })

    # Sort
    if args.sort == "reward":
        results.sort(key=lambda x: x["best_reward"] if x["best_reward"] is not None else float("-inf"),
                     reverse=True)

    # Print
    col_w = max(len(r["folder"]) for r in results) + 2
    header = f"{'Folder':<{col_w}}  {'Best Reward':>14}  {'@ Iter':>7}  {'Rows':>5}"
    print(header)
    print("-" * len(header))

    for r in results:
        if r["best_reward"] is None:
            print(f"{r['folder']:<{col_w}}  {'(no data)':>14}")
        else:
            print(f"{r['folder']:<{col_w}}  {r['best_reward']:>14.4f}  {r['best_iter']:>7}  {r['n_rows']:>5}")


if __name__ == "__main__":
    main()
