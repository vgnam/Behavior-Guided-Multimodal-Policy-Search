"""
Clean up redundant episode directories in log folders.

For each log folder, reads overall_log.txt to determine how many episodes
were actually logged. Any episode_N directories where N >= num_logged_rows
are deleted, since they have no corresponding entry in the overall log.

Usage:
    python cleanup_logs.py [--logs_dir logs] [--dry_run]

By default runs in dry-run mode (only prints what would be deleted).
Use --delete to actually remove directories.
"""

import os
import re
import shutil
import argparse


def cleanup_log_folder(logdir, delete=False):
    """
    Clean a single log folder. Returns (kept, deleted) counts.
    """
    log_path = os.path.join(logdir, "overall_log.txt")
    if not os.path.exists(log_path):
        return 0, 0

    # Count data rows in overall_log.txt (skip header)
    with open(log_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines()[1:] if l.strip()]
    num_logged = len(lines)

    # Find all episode_N directories
    episode_dirs = {}
    for name in os.listdir(logdir):
        m = re.match(r"episode_(\d+)$", name)
        if m and os.path.isdir(os.path.join(logdir, name)):
            episode_dirs[int(m.group(1))] = name

    if not episode_dirs:
        return 0, 0

    # Episodes 0 .. num_logged-1 are valid; anything >= num_logged is redundant
    kept = 0
    deleted = 0
    for ep_num in sorted(episode_dirs.keys()):
        ep_path = os.path.join(logdir, episode_dirs[ep_num])
        if ep_num >= num_logged:
            if delete:
                shutil.rmtree(ep_path)
            deleted += 1
        else:
            kept += 1

    return kept, deleted


def main():
    parser = argparse.ArgumentParser(
        description="Remove episode directories that exceed overall_log.txt row count"
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="logs",
        help="Path to the logs root directory (default: logs)",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete directories. Without this flag, runs in dry-run mode.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.logs_dir):
        print(f"[Error] Logs directory not found: {args.logs_dir}")
        return

    mode = "DELETE" if args.delete else "DRY RUN"
    print(f"Mode: {mode}")
    print(f"Scanning: {args.logs_dir}\n")

    total_kept = 0
    total_deleted = 0

    for folder in sorted(os.listdir(args.logs_dir)):
        folder_path = os.path.join(args.logs_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        log_path = os.path.join(folder_path, "overall_log.txt")
        if not os.path.exists(log_path):
            continue

        # Count logged rows
        with open(log_path, "r", encoding="utf-8") as f:
            num_logged = len([l for l in f.readlines()[1:] if l.strip()])

        # Count episode dirs
        ep_nums = sorted(
            int(re.match(r"episode_(\d+)$", d).group(1))
            for d in os.listdir(folder_path)
            if re.match(r"episode_(\d+)$", d) and os.path.isdir(os.path.join(folder_path, d))
        )

        if not ep_nums:
            continue

        num_dirs = len(ep_nums)
        redundant = [e for e in ep_nums if e >= num_logged]

        if redundant:
            kept, deleted = cleanup_log_folder(folder_path, delete=args.delete)
            action = "Deleted" if args.delete else "Would delete"
            print(f"{folder}")
            print(f"  overall_log rows: {num_logged}, episode dirs: {num_dirs}")
            print(f"  {action} {deleted} dirs (episode_{redundant[0]} .. episode_{redundant[-1]})")
            print(f"  Keeping {kept} dirs (episode_0 .. episode_{num_logged - 1})")
            print()
            total_kept += kept
            total_deleted += deleted
        else:
            print(f"{folder}")
            print(f"  overall_log rows: {num_logged}, episode dirs: {num_dirs} — OK")
            total_kept += num_dirs

    print(f"\nSummary: {total_kept} kept, {total_deleted} {'deleted' if args.delete else 'to delete'}")
    if not args.delete and total_deleted > 0:
        print(f"\nRe-run with --delete to actually remove the {total_deleted} directories.")


if __name__ == "__main__":
    main()
