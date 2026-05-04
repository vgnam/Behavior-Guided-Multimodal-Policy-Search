import yaml
import subprocess
import os
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIGS = [
    ("CliffWalking-CMA-ES-Q", "configs/cliffwalking/cliffwalking_cma_es_qvalue.yaml"),
    ("CliffWalking-OpenAI-ES-Q", "configs/cliffwalking/cliffwalking_openai_es_qvalue.yaml"),
    ("CliffWalking-MuLambda-ES-Q", "configs/cliffwalking/cliffwalking_mu_lambda_es_qvalue.yaml"),
    ("CliffWalking-ARS-Q", "configs/cliffwalking/cliffwalking_ars_qvalue.yaml"),
    ("FrozenLake-CMA-ES-Q", "configs/frozenlake/frozenlake_cma_es_qvalue.yaml"),
    ("FrozenLake-OpenAI-ES-Q", "configs/frozenlake/frozenlake_openai_es_qvalue.yaml"),
    ("FrozenLake-MuLambda-ES-Q", "configs/frozenlake/frozenlake_mu_lambda_es_qvalue.yaml"),
    ("FrozenLake-ARS-Q", "configs/frozenlake/frozenlake_ars_qvalue.yaml"),
]

N_RUNS = 10


def run_single(name, config_rel_path, seed, run_idx):
    config_path = os.path.join(BASE_DIR, config_rel_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config["seed"] = seed
    original_logdir = config["logdir"]
    config["logdir"] = f"{original_logdir}_seed{seed}_run{run_idx}"

    temp_config = os.path.join(
        BASE_DIR, f"temp_{name.replace(' ', '_')}_s{seed}_r{run_idx}.yaml"
    )
    with open(temp_config, "w") as f:
        yaml.dump(config, f)

    try:
        result = subprocess.run(
            ["python", "main.py", "--config", temp_config],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=900,  # 15 minutes per run max
        )
        # Optionally dump stderr/stdout for debugging if needed
        if result.returncode != 0:
            print(f"FAILED: {name} seed={seed} run={run_idx} rc={result.returncode}")
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {name} seed={seed} run={run_idx}")
    finally:
        if os.path.exists(temp_config):
            os.remove(temp_config)

    # Parse best reward from overall_log.txt (max of Total Reward column)
    logdir = os.path.join(BASE_DIR, config["logdir"])
    best_reward = float("-inf")
    overall_log = os.path.join(logdir, "overall_log.txt")
    if os.path.exists(overall_log):
        with open(overall_log, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                next(reader)  # skip header
            except StopIteration:
                pass
            for row in reader:
                if len(row) >= 6:
                    try:
                        reward = float(row[5])
                        if reward > best_reward:
                            best_reward = reward
                    except ValueError:
                        pass

    print(f"DONE: {name} seed={seed} best_reward={best_reward}")
    return name, best_reward


def main():
    results = {name: [] for name, _ in CONFIGS}

    tasks = []
    for name, path in CONFIGS:
        for run_idx in range(N_RUNS):
            seed = run_idx
            tasks.append((name, path, seed, run_idx))

    # Run up to 4 experiments concurrently
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(run_single, name, path, seed, idx)
            for name, path, seed, idx in tasks
        ]
        for future in as_completed(futures):
            try:
                name, best_reward = future.result()
                results[name].append(best_reward)
            except Exception as exc:
                print(f"ERROR during run: {exc}")

    # Print report
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS (Best Mean Reward across 10 runs)")
    print("=" * 80)
    print(f"{'Algorithm':<40s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s} {'N':>4s}")
    print("-" * 80)
    for name, _ in CONFIGS:
        vals = [v for v in results[name] if v != float("-inf")]
        if not vals:
            print(f"{name:<40s} NO DATA")
            continue
        mean = np.mean(vals)
        std = np.std(vals, ddof=1)
        mn = np.min(vals)
        mx = np.max(vals)
        print(f"{name:<40s} {mean:10.3f} {std:10.3f} {mn:10.3f} {mx:10.3f} {len(vals):4d}")
    print("=" * 80)

    # Save JSON
    out_path = os.path.join(BASE_DIR, "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved to: {out_path}")


if __name__ == "__main__":
    main()
