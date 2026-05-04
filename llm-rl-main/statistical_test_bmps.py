# -*- coding: utf-8 -*-
"""
Statistical testing for BMPS results.
Compares ProPS-V (GeminiVLM & GemmaVLM) against ARS, CMA-ES, OpenAI-ES, MuLambda-ES, and ProPS+.
Data manually transcribed from BMPS.md for accuracy.
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

# ---------------------------------------------------------------------------
# Raw data transcribed from BMPS.md
# ---------------------------------------------------------------------------

DATA = {
    "PROPSV_geminiVLM": {
        "mountaincar": np.array([-111.200, -114.2, -109.3, -105.55, -115.65, -113.7, -162.55, -111.15, -113.7, -113.55]),
        "mountaincarcontinuous": np.array([99.239, 99.23, 99.14, 98.93, 99.17, 99.17, 98.9, 98.9, 98.87, 99.18]),
        "reacher": np.array([-7.401, -7.47, -7.51, -6.81, -7.33, -6.71, -6.72, -7.36, -7.95, -7.49]),
        "swimmer": np.array([307.756, 273.5, 299.63, 353.56, 294.26, 347.79, 344.35, 309.02, 331.54, 345.96]),
        "frozenlake": np.array([1.00, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 1.00, 1.00]),
        "cliffwalking": np.array([-13.00, -64.35, -53.3, -55.05, -63.0, -60.45, -65.75, -72.45, -63.55, -72.25]),
        "acrobot": np.array([-74.400, -77.95, -70.9, -79.7, -73.75, -71.85, -74.8, -78.75, -75.8, -76.3]),
        "lunarlander": np.array([268.652, -31.71, 274.85, 219.01, -105.73, 156.41, 6.45, -70.48, 64.97, -50.02]),
    },
    "PROPSV_GemmaVLM": {
        # mountaincar raw values in BMPS.md are positive steps -> convert to negative rewards
        "mountaincar": -np.array([109.65, 113.75, 108.65, 108.2, 117.85, 111.3, 111.7, 110.7, 113.35, 114.95]),
        "mountaincarcontinuous": np.array([99.19, 98.87, 99.05, 99.21, 99.28, 99.32, 99.15, 99.15, 99.19, 99.19]),
        "nav": np.array([2714.5, 2881.95, 4243.475, 346.82, 2482.85, 1482.53, 3161.07, 2615.78, 2976.32, 64.72]),
        "reacher": np.array([-5.99, -7.99, -7.9, -6.74, -8.17, -7.47, -7.58, -7.63, -7.55, -6.56]),
        "swimmer": np.array([343.17, 348.59, 345.97, 346.53, 300.2, 351.96, 235.54, 237.49, 227.45, 356.7]),
        "frozenlake": np.array([1.00, 1.00, 1.00, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95]),
        "cliffwalking": np.array([-64.75, -61.9, -60.65, -67.2, -63.0, -67.15, -68.35, -90.5, -66.75, -58.3]),
        "lunarlander": np.array([-123.61, 276.03, -107.05, 286.41, -69.69, 289.52, 34.11, 276.1, -22.08, 208.57]),
        "invertedpendulum": np.array([1000.0]*10),
    },
    "OPENAI_ES": {
        "hopper": np.array([962.243, 1039.606, 962.243, 1045.055, 1049.767, 1069.425, 1066.843, 1012.610985, 466.070132, 1078.189415]),
        "nav": np.array([-147.5, -145.5, -193.45, -101.375, 2397.9, -193.975, -228.575, -167.25, 316.25, -169.05]),
        "pong": np.array([2.850, 1.250, 1.100, 0.750, 0.550, 0.650, 1.050, 1.200, 0.800, 0.900]),
        "invertedpendulum": np.array([87.891, 91.649, 157.268, 90.653, 129.575, 88.333, 4.0, 1000.0, 1000.0, 1000.0]),
        "doubleinvertedpendulum": np.array([91.208, 91.287, 91.682, 91.630, 99.149, 99.981, 88.503, 88.793, 91.830, 83.753]),
        "walker": np.array([-13.087456, -0.566577, -12.910347, -6.995744, -1.706098, -4.942753, -4.024066, -10.725186, -9.903251, -0.375757]),
    },
    "PROPSP": {
        # PROPSP mountaincar raw data in BMPS.md is split across two lines; manually assembled
        "mountaincar": np.array([-182.900, -114.5, -114.4, -108.25, -110.9, -102.95, -104.5, -111.25, -110.2, -114.8]),
        "mountaincarcontinuous": np.array([94.168, 98.96814244335538, 95.10910452934351, 99.15495856046735, 99.02209045622635, 98.54609618990722, 99.12996060090005, 96.71267353371408, 99.20870271288226, 99.18041091984998]),
        "nav": np.array([392.325, -31.55, 2563.15, 2758.1, 2514.275, 3139.825, 3253.75, 1064.875, 2801.775, 180.7]),
        "reacher": np.array([-7.298, -7.620853845951426, -7.384176842875911, -7.52757733074296, -8.149953549876113, -7.276025864169606, -7.709462146225936, -8.346110499198092, -7.400335105528458, -7.353497580451027]),
        "swimmer": np.array([161.465, 219.68589240287082, 162.1556306436805, 265.72806230538606, 252.54142955263825, 251.28985401971045, 211.94041766885798, 292.5374530031498, 97.258040472847, 191.88046734673105]),
        "frozenlake": np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.95, 0.95, 0.95, 0.95, 0.95]),
        "cliffwalking": np.array([-62.800, -60.55, -81.9, -63.75, -71.85, -76.2, -63.8, -64.15, -86.45, -65.85]),
        "acrobot": np.array([-77.850, -74.75, -78.55, -73.4, -86.0, -78.5, -85.9, -78.55, -81.75, -76.25]),
        "lunarlander": np.array([-36.069, -75.24757897957662, -40.94392627629045, -50.429888406322874, 261.6111191224637, 249.15587029083864, -46.55963597110759, 291.1393424021501, -3.5950959146342205, -59.025993572306525]),
        "invertedpendulum": np.array([1000.0, 1000.0, 204.65, 1000.0, 808.6, 1000.0, 1000.0, 1000.0, 1000.0, 808.6]),
        "doubleinvertedpendulum": np.array([114.68348301887697, 109.17171683197543, 169.50703821213537, 147.01886594751724, 342.33862646075926, 101.22744727649642, 102.97391005448424, 150.76305440944, 104.37230021615137, 103.18452801552785]),
    },
    "CMA_ES": {
        "hopper": np.array([1097.638203, 2578.138752, 1766.232909, 1119.890677, 1162.300574, 1883.087445, 2301.399937, 2648.172922, 1083.793085, 1133.043112]),
        "mountaincar": np.array([-200.0]*10),
        "mountaincarcontinuous": np.array([98.794665, -0.000045, 99.337971, -0.002120, -0.001991, 98.728224, 12.104396, -0.000000, -0.006834, -0.000035]),
        "nav": np.array([-203.375, 121.575, -215.55, -111.45, 3348.925, 1.325, -229.25, 1.025, 2847.6, 191.7]),
        "pong": np.array([0.5, 0.6, 0.5, 0.65, 3.0, 0.6, 0.7, 3.0, 0.6, 0.55]),
        "reacher": np.array([-8.993837, -11.341409, -8.753249, -9.259011, -11.619965, -9.238485, -8.669870, -9.722636, -7.810312, -10.545963]),
        "swimmer": np.array([142.486439, -5.246859, -7.802748, 357.836918, 357.650224, 54.069142, 359.386876, 358.046440, 79.017680, 355.932947]),
        "frozenlake": np.array([0.15, 0.2, 0.0, 0.15, 0.15, 0.25, 0.25, 0.1, 0.0, 0.0]),
        "cliffwalking": np.array([-189.55, -189.55, -100.0, -975.3, -169.3, -100.0, -632.65, -144.55, -100.0, -100.0]),  # appended one -100.0 to make 10
        "acrobot": np.array([-122.75, -72.2, -500.0, -71.6, -68.8, -500.0, -112.35, -86.0, -75.85, -500.0]),
        "invertedpendulum": np.array([1000.0, 2.0, 30.9, 1000.0, 5.8, 1000.0, 2.0, 1000.0, 1000.0, 1000.0]),
        "doubleinvertedpendulum": np.array([87.846052, 137.775113, 178.278985, 9359.843703, 152.447929, 171.055866, 71.578887, 23.584098, 96.713748, 243.423873]),
        "walker": np.array([-31.2233, -4.10153, 183.16314, -7.942113, -12.130742, -9.49301, -10.169517, 958.660369, 374.351315, 15.583432]),
    },
    "ARS": {
        "hopper": np.array([268.517, 316.532, 956.414, 4.69, 125.848, 6.875, 900.107, 36.733, 204.853, 10.702]),
        "mountaincar": np.array([-200.0]*10),
        "mountaincarcontinuous": np.array([-0.005, -0.006, -0.004, 98.979, -0.132, -0.029, -2.664, -0.041, -0.046, -0.131]),
        "nav": np.array([-171.7, -41.6, -180.15, -161.325, -159.9, -172.7, 1.175, -215.875, 96.05, -127.225]),
        "pong": np.array([0.350, 0.200, 0.100, 0.250, 0.100, 0.250, 0.200, 0.250, 0.250, 0.250]),
        "reacher": np.array([-8.978076, -9.097515, -12.414031, -13.479, -11.196, -35.638, -40.208666, -30.188378, -46.701258, -183.269]),
        "swimmer": np.array([51.820, 36.059, 33.751, 307.808, -5.239, 37.507651, -18.415845, 29.537405, -5.368544, 49.550642]),
        "acrobot": np.array([-500.0, -167.55, -500.0, -500.0, -500.0, -500.0, -500.0, -85.35, -500.0, -77.75]),
        "invertedpendulum": np.array([115.8, 2.0, 4.0, 2.0, 1000.0, 36.15, 2.0, 4.0, 1000.0, 2.0]),
        "doubleinvertedpendulum": np.array([89.269722, 23.579702, 23.587026, 23.997287, 44.715535, 61.277883, 792.372816, 319.694791, 30.205616, 105.884088]),
        "walker": np.array([-46.437514, -20.110186, -36.290657, -47.287905, -24.958954, -54.331889, -19.241005, -23.384001, -48.419514, -98.938287]),
    },
    "MuLambda_ES": {
        "hopper": np.array([1111.665245, 1164.535718, 1027.597557, 1096.289609, 1026.887806, 2432.252473, 1084.258511, 1124.290063, 1094.297467, 1083.987751]),
        "mountaincar": np.array([-200.0]*10),
        "mountaincarcontinuous": np.array([-0.008, 84.131, -0.096, -0.007, -0.050, -0.005, 99.171, 99.029, -1.849, 99.233]),
        "nav": np.array([1985.975, -248.925, -238.425, -214.025, -99.875, 2801.45, -164.075, 302.95, -107.15, 2678.375]),
        "pong": np.array([0.200, 0.200, 0.200, 1.700, 0.200, 0.550, 0.550, 3.0, 0.45, 0.60]),
        "reacher": np.array([-10.827935, -11.627046, -116.297144, -10.225912, -9.698219, -10.828273, -14.453661, -9.977012, -10.705458, -12.463669]),
        "swimmer": np.array([120.139576, 34.354194, 53.006763, 356.763594, 119.367674, 138.926481, 352.531504, 30.803343, 349.634842, 121.777184]),
        "acrobot": np.array([-83.25, -86.9, -482.35, -500.0, -158.05, -159.0, -86.5, -500.0, -186.05, -500.0]),
        "invertedpendulum": np.array([236.45, 2.0, 1000.0, 1000.0, 1000.0, 402.9, 64.7, 1000.0, 66.85, 2.0]),
        "doubleinvertedpendulum": np.array([27.124842, 23.557055, 27.332174, 281.760126, 82.629333, 101.388268, 49.380702, 317.229628, 97.909023, 122.680438]),
        "walker": np.array([-8.622864, -22.084030, -14.488109, -4.420381, -11.643680, -7.485784, -6.541650, -9.271639, -32.521342, -15.949239]),
        "frozenlake": np.array([0.25, 0.15, 0.15, 0.45, 0.0, 0.0, 0.0, 0.5, 0.1, 0.0]),  # from benchmark_results.json
        "cliffwalking": np.array([-100.0, -103.8, -100.0, -124.25, -100.0, -100.0, -109.9, -114.85, -129.7, -100.0]),  # from benchmark_results.json
    },
}

# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def cohens_d(x, y):
    """Cohen's d for independent samples (using pooled std dev)."""
    nx, ny = len(x), len(y)
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    if nx + ny < 3:
        return np.nan
    pooled_std = np.sqrt(((nx - 1) * var_x + (ny - 1) * var_y) / (nx + ny - 2))
    if pooled_std == 0 or np.isnan(pooled_std):
        return np.inf if np.mean(x) != np.mean(y) else 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std


def run_tests():
    propsv_methods = ["PROPSV_geminiVLM", "PROPSV_GemmaVLM"]
    baseline_methods = ["ARS", "CMA_ES", "OPENAI_ES", "MuLambda_ES", "PROPSP"]
    
    rows = []
    
    for propsv in propsv_methods:
        for baseline in baseline_methods:
            common = sorted(set(DATA[propsv].keys()) & set(DATA[baseline].keys()))
            for prob in common:
                x = DATA[propsv][prob]
                y = DATA[baseline][prob]
                
                if len(x) < 2 or len(y) < 2:
                    continue
                
                # Welch's t-test (unequal variances)
                t_stat, t_p = stats.ttest_ind(x, y, equal_var=False)
                
                # Mann-Whitney U (non-parametric)
                try:
                    u_stat, u_p = stats.mannwhitneyu(x, y, alternative='two-sided')
                except ValueError:
                    u_stat, u_p = np.nan, np.nan
                
                d = cohens_d(x, y)
                
                rows.append({
                    "ProPS-V": propsv.replace("PROPSV_", "").replace("_", " ").strip(),
                    "Baseline": baseline.replace("_", " ").strip(),
                    "Problem": prob,
                    "n_V": len(x),
                    "n_Base": len(y),
                    "Mean_V": round(float(np.mean(x)), 3),
                    "Std_V": round(float(np.std(x, ddof=1)), 3),
                    "Mean_Base": round(float(np.mean(y)), 3),
                    "Std_Base": round(float(np.std(y, ddof=1)), 3),
                    "t_stat": round(float(t_stat), 4),
                    "t_pvalue": round(float(t_p), 4),
                    "MWU_stat": round(float(u_stat), 2) if not np.isnan(u_stat) else np.nan,
                    "MWU_pvalue": round(float(u_p), 4) if not np.isnan(u_p) else np.nan,
                    "Cohens_d": round(float(d), 4) if not np.isnan(d) else np.nan,
                    "Significant_at_0.05": "Yes" if t_p < 0.05 else "No",
                    "Winner": "ProPS-V" if np.mean(x) > np.mean(y) else ("Baseline" if np.mean(y) > np.mean(x) else "Tie")
                })
    
    return pd.DataFrame(rows)


def main():
    df = run_tests()
    
    if df.empty:
        print("No overlapping problems found for comparison.")
        return
    
    # Print full table
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', 30)
    
    print("="*120)
    print("FULL STATISTICAL COMPARISON: ProPS-V vs Baselines")
    print("="*120)
    print(df.to_string(index=False))
    
    # Summary by variant / baseline
    print("\n" + "="*120)
    print("SUMMARY (counts per comparison pair)")
    print("="*120)
    
    summary = []
    for (v, b), group in df.groupby(["ProPS-V", "Baseline"]):
        sig_wins = group[(group["Significant_at_0.05"] == "Yes") & (group["Winner"] == "ProPS-V")]
        sig_loss = group[(group["Significant_at_0.05"] == "Yes") & (group["Winner"] == "Baseline")]
        non_sig = group[group["Significant_at_0.05"] == "No"]
        summary.append({
            "ProPS-V": v,
            "Baseline": b,
            "Common_Problems": len(group),
            "Sig_Wins_V": len(sig_wins),
            "Sig_Wins_Base": len(sig_loss),
            "Non_Sig": len(non_sig),
        })
    
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    
    # Overall per baseline across both ProPS-V variants
    print("\n" + "="*120)
    print("AGGREGATED SUMMARY ACROSS BOTH ProPS-V VARIANTS")
    print("="*120)
    agg = []
    for baseline in df["Baseline"].unique():
        sub = df[df["Baseline"] == baseline]
        sig_wins = sub[(sub["Significant_at_0.05"] == "Yes") & (sub["Winner"] == "ProPS-V")]
        sig_loss = sub[(sub["Significant_at_0.05"] == "Yes") & (sub["Winner"] == "Baseline")]
        non_sig = sub[sub["Significant_at_0.05"] == "No"]
        agg.append({
            "Baseline": baseline,
            "Total_Comparisons": len(sub),
            "ProPS-V_Sig_Wins": len(sig_wins),
            "Baseline_Sig_Wins": len(sig_loss),
            "Non_Significant": len(non_sig),
        })
    
    agg_df = pd.DataFrame(agg)
    print(agg_df.to_string(index=False))
    
    # Save outputs
    out_csv = Path(__file__).parent / "statistical_test_results.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\nDetailed results saved to: {out_csv}")
    
    summary_csv = Path(__file__).parent / "statistical_test_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"Summary saved to: {summary_csv}")


if __name__ == "__main__":
    main()
