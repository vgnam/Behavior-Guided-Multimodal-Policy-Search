# -*- coding: utf-8 -*-
"""
Plot CD Diagram and Win-Tie-Loss Heatmap for BMPS results.

Requires: numpy, pandas, scipy, matplotlib, seaborn
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Use non-interactive backend to avoid display issues
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Raw data from BMPS.md (mean of 10 runs per problem)
# ---------------------------------------------------------------------------

RAW_DATA = {
    "PROPSV_geminiVLM": {
        "mountaincar": [-111.200, -114.2, -109.3, -105.55, -115.65, -113.7, -162.55, -111.15, -113.7, -113.55],
        "mountaincarcontinuous": [99.239, 99.23, 99.14, 98.93, 99.17, 99.17, 98.9, 98.9, 98.87, 99.18],
        "reacher": [-7.401, -7.47, -7.51, -6.81, -7.33, -6.71, -6.72, -7.36, -7.95, -7.49],
        "swimmer": [307.756, 273.5, 299.63, 353.56, 294.26, 347.79, 344.35, 309.02, 331.54, 345.96],
        "frozenlake": [1.00, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 1.00, 1.00],
        "cliffwalking": [-13.00, -64.35, -53.3, -55.05, -63.0, -60.45, -65.75, -72.45, -63.55, -72.25],
        "acrobot": [-74.400, -77.95, -70.9, -79.7, -73.75, -71.85, -74.8, -78.75, -75.8, -76.3],
        "lunarlander": [268.652, -31.71, 274.85, 219.01, -105.73, 156.41, 6.45, -70.48, 64.97, -50.02],
    },
    "PROPSV_GemmaVLM": {
        "mountaincar": [-109.65, -113.75, -108.65, -108.2, -117.85, -111.3, -111.7, -110.7, -113.35, -114.95],
        "mountaincarcontinuous": [99.19, 98.87, 99.05, 99.21, 99.28, 99.32, 99.15, 99.15, 99.19, 99.19],
        "nav": [2714.5, 2881.95, 4243.475, 346.82, 2482.85, 1482.53, 3161.07, 2615.78, 2976.32, 64.72],
        "reacher": [-5.99, -7.99, -7.9, -6.74, -8.17, -7.47, -7.58, -7.63, -7.55, -6.56],
        "swimmer": [343.17, 348.59, 345.97, 346.53, 300.2, 351.96, 235.54, 237.49, 227.45, 356.7],
        "frozenlake": [1.00, 1.00, 1.00, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95],
        "cliffwalking": [-64.75, -61.9, -60.65, -67.2, -63.0, -67.15, -68.35, -90.5, -66.75, -58.3],
        "lunarlander": [-123.61, 276.03, -107.05, 286.41, -69.69, 289.52, 34.11, 276.1, -22.08, 208.57],
        "invertedpendulum": [1000.0]*10,
    },
    "OPENAI_ES": {
        "hopper": [962.243, 1039.606, 962.243, 1045.055, 1049.767, 1069.425, 1066.843, 1012.610985, 466.070132, 1078.189415],
        "nav": [-147.5, -145.5, -193.45, -101.375, 2397.9, -193.975, -228.575, -167.25, 316.25, -169.05],
        "pong": [2.850, 1.250, 1.100, 0.750, 0.550, 0.650, 1.050, 1.200, 0.800, 0.900],
        "invertedpendulum": [87.891, 91.649, 157.268, 90.653, 129.575, 88.333, 4.0, 1000.0, 1000.0, 1000.0],
        "doubleinvertedpendulum": [91.208, 91.287, 91.682, 91.630, 99.149, 99.981, 88.503, 88.793, 91.830, 83.753],
        "walker": [-13.087456, -0.566577, -12.910347, -6.995744, -1.706098, -4.942753, -4.024066, -10.725186, -9.903251, -0.375757],
    },
    "PROPSP": {
        "mountaincar": [-182.900, -114.5, -114.4, -108.25, -110.9, -102.95, -104.5, -111.25, -110.2, -114.8],
        "mountaincarcontinuous": [94.168, 98.96814244335538, 95.10910452934351, 99.15495856046735, 99.02209045622635, 98.54609618990722, 99.12996060090005, 96.71267353371408, 99.20870271288226, 99.18041091984998],
        "nav": [392.325, -31.55, 2563.15, 2758.1, 2514.275, 3139.825, 3253.75, 1064.875, 2801.775, 180.7],
        "reacher": [-7.298, -7.620853845951426, -7.384176842875911, -7.52757733074296, -8.149953549876113, -7.276025864169606, -7.709462146225936, -8.346110499198092, -7.400335105528458, -7.353497580451027],
        "swimmer": [161.465, 219.68589240287082, 162.1556306436805, 265.72806230538606, 252.54142955263825, 251.28985401971045, 211.94041766885798, 292.5374530031498, 97.258040472847, 191.88046734673105],
        "frozenlake": [0.9, 0.9, 0.9, 0.9, 0.9, 0.95, 0.95, 0.95, 0.95, 0.95],
        "cliffwalking": [-62.800, -60.55, -81.9, -63.75, -71.85, -76.2, -63.8, -64.15, -86.45, -65.85],
        "acrobot": [-77.850, -74.75, -78.55, -73.4, -86.0, -78.5, -85.9, -78.55, -81.75, -76.25],
        "lunarlander": [-36.069, -75.24757897957662, -40.94392627629045, -50.429888406322874, 261.6111191224637, 249.15587029083864, -46.55963597110759, 291.1393424021501, -3.5950959146342205, -59.025993572306525],
        "invertedpendulum": [1000.0, 1000.0, 204.65, 1000.0, 808.6, 1000.0, 1000.0, 1000.0, 1000.0, 808.6],
        "doubleinvertedpendulum": [114.68348301887697, 109.17171683197543, 169.50703821213537, 147.01886594751724, 342.33862646075926, 101.22744727649642, 102.97391005448424, 150.76305440944, 104.37230021615137, 103.18452801552785],
    },
    "CMA_ES": {
        "hopper": [1097.638203, 2578.138752, 1766.232909, 1119.890677, 1162.300574, 1883.087445, 2301.399937, 2648.172922, 1083.793085, 1133.043112],
        "mountaincar": [-200.0]*10,
        "mountaincarcontinuous": [98.794665, -0.000045, 99.337971, -0.002120, -0.001991, 98.728224, 12.104396, -0.000000, -0.006834, -0.000035],
        "nav": [-203.375, 121.575, -215.55, -111.45, 3348.925, 1.325, -229.25, 1.025, 2847.6, 191.7],
        "pong": [0.5, 0.6, 0.5, 0.65, 3.0, 0.6, 0.7, 3.0, 0.6, 0.55],
        "reacher": [-8.993837, -11.341409, -8.753249, -9.259011, -11.619965, -9.238485, -8.669870, -9.722636, -7.810312, -10.545963],
        "swimmer": [142.486439, -5.246859, -7.802748, 357.836918, 357.650224, 54.069142, 359.386876, 358.046440, 79.017680, 355.932947],
        "frozenlake": [0.15, 0.2, 0.0, 0.15, 0.15, 0.25, 0.25, 0.1, 0.0, 0.0],
        "cliffwalking": [-189.55, -189.55, -100.0, -975.3, -169.3, -100.0, -632.65, -144.55, -100.0, -100.0],
        "acrobot": [-122.75, -72.2, -500.0, -71.6, -68.8, -500.0, -112.35, -86.0, -75.85, -500.0],
        "invertedpendulum": [1000.0, 2.0, 30.9, 1000.0, 5.8, 1000.0, 2.0, 1000.0, 1000.0, 1000.0],
        "doubleinvertedpendulum": [87.846052, 137.775113, 178.278985, 9359.843703, 152.447929, 171.055866, 71.578887, 23.584098, 96.713748, 243.423873],
        "walker": [-31.2233, -4.10153, 183.16314, -7.942113, -12.130742, -9.49301, -10.169517, 958.660369, 374.351315, 15.583432],
    },
    "ARS": {
        "hopper": [268.517, 316.532, 956.414, 4.69, 125.848, 6.875, 900.107, 36.733, 204.853, 10.702],
        "mountaincar": [-200.0]*10,
        "mountaincarcontinuous": [-0.005, -0.006, -0.004, 98.979, -0.132, -0.029, -2.664, -0.041, -0.046, -0.131],
        "nav": [-171.7, -41.6, -180.15, -161.325, -159.9, -172.7, 1.175, -215.875, 96.05, -127.225],
        "pong": [0.350, 0.200, 0.100, 0.250, 0.100, 0.250, 0.200, 0.250, 0.250, 0.250],
        "reacher": [-8.978076, -9.097515, -12.414031, -13.479, -11.196, -35.638, -40.208666, -30.188378, -46.701258, -183.269],
        "swimmer": [51.820, 36.059, 33.751, 307.808, -5.239, 37.507651, -18.415845, 29.537405, -5.368544, 49.550642],
        "acrobot": [-500.0, -167.55, -500.0, -500.0, -500.0, -500.0, -500.0, -85.35, -500.0, -77.75],
        "invertedpendulum": [115.8, 2.0, 4.0, 2.0, 1000.0, 36.15, 2.0, 4.0, 1000.0, 2.0],
        "doubleinvertedpendulum": [89.269722, 23.579702, 23.587026, 23.997287, 44.715535, 61.277883, 792.372816, 319.694791, 30.205616, 105.884088],
        "walker": [-46.437514, -20.110186, -36.290657, -47.287905, -24.958954, -54.331889, -19.241005, -23.384001, -48.419514, -98.938287],
    },
    "MuLambda_ES": {
        "hopper": [1111.665245, 1164.535718, 1027.597557, 1096.289609, 1026.887806, 2432.252473, 1084.258511, 1124.290063, 1094.297467, 1083.987751],
        "mountaincar": [-200.0]*10,
        "mountaincarcontinuous": [-0.008, 84.131, -0.096, -0.007, -0.050, -0.005, 99.171, 99.029, -1.849, 99.233],
        "nav": [1985.975, -248.925, -238.425, -214.025, -99.875, 2801.45, -164.075, 302.95, -107.15, 2678.375],
        "pong": [0.200, 0.200, 0.200, 1.700, 0.200, 0.550, 0.550, 3.0, 0.45, 0.60],
        "reacher": [-10.827935, -11.627046, -116.297144, -10.225912, -9.698219, -10.828273, -14.453661, -9.977012, -10.705458, -12.463669],
        "swimmer": [120.139576, 34.354194, 53.006763, 356.763594, 119.367674, 138.926481, 352.531504, 30.803343, 349.634842, 121.777184],
        "acrobot": [-83.25, -86.9, -482.35, -500.0, -158.05, -159.0, -86.5, -500.0, -186.05, -500.0],
        "invertedpendulum": [236.45, 2.0, 1000.0, 1000.0, 1000.0, 402.9, 64.7, 1000.0, 66.85, 2.0],
        "doubleinvertedpendulum": [27.124842, 23.557055, 27.332174, 281.760126, 82.629333, 101.388268, 49.380702, 317.229628, 97.909023, 122.680438],
        "walker": [-8.622864, -22.084030, -14.488109, -4.420381, -11.643680, -7.485784, -6.541650, -9.271639, -32.521342, -15.949239],
        "frozenlake": [0.25, 0.15, 0.15, 0.45, 0.0, 0.0, 0.0, 0.5, 0.1, 0.0],
        "cliffwalking": [-100.0, -103.8, -100.0, -124.25, -100.0, -100.0, -109.9, -114.85, -129.7, -100.0],
    },
}

METHOD_LABELS = {
    "PROPSV_geminiVLM": "ProPS-V (Gemini)",
    "PROPSV_GemmaVLM": "ProPS-V (Gemma)",
    "OPENAI_ES": "OpenAI-ES",
    "PROPSP": "ProPS+",
    "CMA_ES": "CMA-ES",
    "ARS": "ARS",
    "MuLambda_ES": "MuLambda-ES",
}

# ---------------------------------------------------------------------------
# Build problem x method mean reward matrix (for Friedman)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CD Diagram (uses subset of methods with most common problems)
# ---------------------------------------------------------------------------

def build_mean_matrix_for_cd():
    """
    Build DataFrame for CD diagram using the largest subset of methods
    that share the most common problems.
    """
    # We use 5 main methods that have the most overlap (exclude ARS to gain more common problems):
    cd_methods = ["PROPSV_geminiVLM", "PROPSV_GemmaVLM", "PROPSP", "CMA_ES", "MuLambda_ES"]
    
    # Find problems common to all 5 methods
    common_problems = set(RAW_DATA[cd_methods[0]].keys())
    for m in cd_methods[1:]:
        common_problems &= set(RAW_DATA[m].keys())
    
    common_problems = sorted(common_problems)
    print(f"CD Diagram uses {len(common_problems)} common problems: {common_problems}")
    
    records = []
    for problem in common_problems:
        row = {"Problem": problem}
        for method in cd_methods:
            row[method] = np.mean(RAW_DATA[method][problem])
        records.append(row)
    
    df = pd.DataFrame(records)
    return df, cd_methods


def compute_ranks(df, methods):
    """Compute average rank per method across problems (higher reward = rank 1)."""
    problem_cols = methods
    
    # Rank per problem: rank 1 = best (highest reward)
    rank_matrix = df[problem_cols].rank(axis=1, ascending=False, method="average")
    avg_ranks = rank_matrix.mean(axis=0).sort_values()
    
    return avg_ranks, rank_matrix


def nemenyi_cd(num_datasets, num_methods, alpha=0.05):
    """Critical difference for Nemenyi test (approximate q_alpha from table)."""
    # Common q_alpha values for Nemenyi at alpha=0.05
    q_alpha_table = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
        7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164
    }
    q_alpha = q_alpha_table.get(num_methods, 2.728)
    cd = q_alpha * np.sqrt(num_methods * (num_methods + 1) / (6.0 * num_datasets))
    return cd


def plot_cd_diagram(avg_ranks, cd, output_path):
    """Draw a horizontal CD diagram."""
    methods = list(avg_ranks.index)
    ranks = list(avg_ranks.values)
    n_methods = len(methods)
    
    fig, ax = plt.subplots(figsize=(10, 3 + n_methods * 0.4))
    
    # Sort by rank
    order = np.argsort(ranks)
    methods = [methods[i] for i in order]
    ranks = [ranks[i] for i in order]
    
    # y positions
    y_pos = np.arange(n_methods)
    
    # Draw axis
    ax.hlines(0, 1, n_methods, colors="black", linewidth=1)
    for i, (m, r) in enumerate(zip(methods, ranks)):
        ax.plot(r, 0, "k|", markersize=10)
        ax.text(r, -0.15, f"{r:.2f}", ha="center", va="top", fontsize=9)
        # Method name on right
        ax.text(n_methods + 0.3, i, METHOD_LABELS.get(m, m), va="center", ha="left", fontsize=10, fontweight="bold")
        # Dot
        ax.plot(r, i, "o", markersize=10, color=plt.cm.tab10(i))
        # Line from axis to dot
        ax.plot([r, r], [0, i - 0.15], color="gray", linewidth=0.8, linestyle="--")
    
    # Draw CD bar at top
    left_cd = 1.0
    ax.hlines(n_methods - 0.5, left_cd, left_cd + cd, colors="red", linewidth=3)
    ax.text(left_cd + cd / 2, n_methods - 0.35, f"CD = {cd:.2f}", ha="center", va="bottom", color="red", fontsize=10, fontweight="bold")
    
    # Group connected methods (no significant difference)
    # Simple greedy grouping
    groups = []
    visited = set()
    for i in range(n_methods):
        if i in visited:
            continue
        group = [i]
        for j in range(i + 1, n_methods):
            if abs(ranks[i] - ranks[j]) <= cd:
                group.append(j)
                visited.add(j)
        groups.append(group)
        visited.add(i)
    
    # Draw brackets for groups
    bracket_y = -0.6
    for g in groups:
        if len(g) > 1:
            min_r = min(ranks[k] for k in g)
            max_r = max(ranks[k] for k in g)
            ax.plot([min_r, max_r], [bracket_y, bracket_y], color="black", linewidth=2)
            ax.plot([min_r, min_r], [bracket_y, bracket_y + 0.1], color="black", linewidth=2)
            ax.plot([max_r, max_r], [bracket_y, bracket_y + 0.1], color="black", linewidth=2)
            bracket_y -= 0.25
    
    ax.set_xlim(0.5, n_methods + 2.5)
    ax.set_ylim(bracket_y - 0.3, n_methods + 0.5)
    ax.set_yticks([])
    ax.set_xticks(range(1, n_methods + 1))
    ax.set_xticklabels([str(i) for i in range(1, n_methods + 1)])
    ax.set_xlabel("Average Rank (lower is better)", fontsize=12)
    ax.set_title("Critical Difference Diagram (Friedman + Nemenyi, α = 0.05)", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"CD Diagram saved to: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Win-Tie-Loss Heatmap
# ---------------------------------------------------------------------------

def build_wtl_matrix():
    """
    Build Win-Tie-Loss matrix between ProPS-V variants and baselines.
    Uses Welch's t-test (alpha=0.05) on each common problem.
    Returns DataFrames for wins, ties, losses counts.
    """
    propsv_methods = ["PROPSV_geminiVLM", "PROPSV_GemmaVLM"]
    baseline_methods = ["ARS", "CMA_ES", "OPENAI_ES", "MuLambda_ES", "PROPSP"]
    alpha = 0.05
    
    wins = []
    ties = []
    losses = []
    
    for propsv in propsv_methods:
        w_row = []
        t_row = []
        l_row = []
        for baseline in baseline_methods:
            common = sorted(set(RAW_DATA[propsv].keys()) & set(RAW_DATA[baseline].keys()))
            w, t, l = 0, 0, 0
            for prob in common:
                x = np.array(RAW_DATA[propsv][prob])
                y = np.array(RAW_DATA[baseline][prob])
                if len(x) < 2 or len(y) < 2:
                    continue
                _, p = stats.ttest_ind(x, y, equal_var=False)
                if p >= alpha:
                    t += 1
                else:
                    if np.mean(x) > np.mean(y):
                        w += 1
                    else:
                        l += 1
            w_row.append(w)
            t_row.append(t)
            l_row.append(l)
        wins.append(w_row)
        ties.append(t_row)
        losses.append(l_row)
    
    index = [METHOD_LABELS[m] for m in propsv_methods]
    columns = [METHOD_LABELS[m] for m in baseline_methods]
    
    wins_df = pd.DataFrame(wins, index=index, columns=columns)
    ties_df = pd.DataFrame(ties, index=index, columns=columns)
    losses_df = pd.DataFrame(losses, index=index, columns=columns)
    
    return wins_df, ties_df, losses_df


def plot_wtl_heatmap(wins_df, ties_df, losses_df, output_path):
    """Plot a combined Win-Tie-Loss annotated heatmap."""
    # Create a combined annotation matrix: "W/T/L"
    annot = []
    for i in range(wins_df.shape[0]):
        row = []
        for j in range(wins_df.shape[1]):
            w = wins_df.iloc[i, j]
            t = ties_df.iloc[i, j]
            l = losses_df.iloc[i, j]
            row.append(f"{w}/{t}/{l}")
        annot.append(row)
    annot = np.array(annot)
    
    # Compute a composite score for coloring: Win - Loss (higher = better for ProPS-V)
    composite = wins_df.values - losses_df.values
    
    fig, ax = plt.subplots(figsize=(8, 3.5))
    
    cmap = sns.diverging_palette(10, 133, s=85, l=55, n=9, center="light", as_cmap=True)
    sns.heatmap(
        composite,
        annot=annot,
        fmt="",
        cmap=cmap,
        center=0,
        linewidths=1,
        linecolor="white",
        square=True,
        cbar_kws={"label": "Wins - Losses"},
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
    )
    
    ax.set_xlabel("Baseline Method", fontsize=12, fontweight="bold")
    ax.set_ylabel("ProPS-V Variant", fontsize=12, fontweight="bold")
    ax.set_title("Win / Tie / Loss Counts (α = 0.05, Welch's t-test)", fontsize=13, fontweight="bold")
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Win-Tie-Loss Heatmap saved to: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = Path(__file__).parent
    
    # --- CD Diagram ---
    df_mean, cd_methods = build_mean_matrix_for_cd()
    avg_ranks, rank_matrix = compute_ranks(df_mean, cd_methods)
    
    num_datasets = len(df_mean)
    num_methods = len(avg_ranks)
    cd = nemenyi_cd(num_datasets, num_methods, alpha=0.05)
    
    print(f"CD Diagram: {num_datasets} common problems, {num_methods} methods")
    print(f"Critical Difference (Nemenyi, alpha=0.05): {cd:.4f}")
    print("\nAverage Ranks:")
    for m, r in avg_ranks.items():
        print(f"  {METHOD_LABELS.get(m, m):20s}: {r:.3f}")
    
    plot_cd_diagram(avg_ranks, cd, out_dir / "cd_diagram.png")
    
    # --- Win-Tie-Loss Heatmap ---
    wins_df, ties_df, losses_df = build_wtl_matrix()
    print("\nWin-Tie-Loss Matrix:")
    print(wins_df)
    print(ties_df)
    print(losses_df)
    
    plot_wtl_heatmap(wins_df, ties_df, losses_df, out_dir / "win_tie_loss_heatmap.png")


if __name__ == "__main__":
    main()
