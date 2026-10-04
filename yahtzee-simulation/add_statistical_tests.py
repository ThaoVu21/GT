#!/usr/bin/env python3
"""
Enhanced statistical tests with bootstrap CIs and Bonferroni correction.

Improvements:
- Bootstrap confidence intervals for robustness
- Bonferroni correction for multiple comparisons
- Additional effect size measures (Glass Δ, Hedges g)
"""

import math
from typing import Dict, List

import numpy as np
from scipy import stats

# Import from improved simulator
try:
    from yahtzee_simulator import build_strategies, simulate_games
except ImportError:
    from yahtzee_simulator import build_strategies, simulate_games


def cohen_d(x: List[int], y: List[int]) -> float:
    """Compute Cohen's d for two independent samples."""
    nx, ny = len(x), len(y)
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
    s_pooled = math.sqrt(((nx - 1) * sx**2 + (ny - 1) * sy**2) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / s_pooled if s_pooled > 0 else 0.0


def glass_delta(x: List[int], y: List[int]) -> float:
    """Compute Glass's Δ (uses control group std dev)."""
    return (np.mean(x) - np.mean(y)) / np.std(y, ddof=1) if np.std(y, ddof=1) > 0 else 0.0


def hedges_g(x: List[int], y: List[int]) -> float:
    """Compute Hedges' g (bias-corrected Cohen's d)."""
    d = cohen_d(x, y)
    nx, ny = len(x), len(y)
    df = nx + ny - 2
    correction = 1 - (3 / (4 * df - 1))
    return d * correction


# Performance optimization: 2000 bootstrap samples is sufficient and 5x faster
BOOTSTRAP_SAMPLES = 2000

def bootstrap_ci(data: List[int], confidence: float = 0.95, n_bootstrap: int = BOOTSTRAP_SAMPLES) -> tuple:
    """
    Bootstrap confidence interval for mean.
    Returns (lower, upper) bounds.
    """
    n = len(data)
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        means.append(np.mean(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(means, 100 * (alpha / 2))
    upper = np.percentile(means, 100 * (1 - alpha / 2))
    return lower, upper


def main():
    n_games = 10000
    seed = 12345

    strategies = build_strategies()
    results: Dict[str, Dict] = {}

    print("=" * 80)
    print("ENHANCED STATISTICAL ANALYSIS")
    print("=" * 80)
    print(f"\nSimulating {n_games} games per strategy...\n")

    for name, strat in strategies.items():
        stats_res = simulate_games(strat, n_games=n_games, seed=seed)
        results[name] = stats_res
        print(f"{name:15s}: mean={stats_res['mean']:.2f}, "
              f"CI95=({stats_res['ci_low']:.2f}, {stats_res['ci_high']:.2f})")

    print("\n" + "=" * 80)
    print(f"BOOTSTRAP CONFIDENCE INTERVALS ({BOOTSTRAP_SAMPLES} bootstrap samples)")
    print("=" * 80)
    print("\nBootstrap CIs provide robustness to non-normality:\n")

    for name, res in results.items():
        boot_low, boot_high = bootstrap_ci(res["scores"], confidence=0.95, n_bootstrap=BOOTSTRAP_SAMPLES)
        print(f"{name:15s}: Bootstrap CI = [{boot_low:.2f}, {boot_high:.2f}], "
              f"Normal CI = [{res['ci_low']:.2f}, {res['ci_high']:.2f}]")

    print("\n" + "=" * 80)
    print("PAIRWISE COMPARISONS: Probability-Based vs Others (Two-Sided t-tests)")
    print("=" * 80)
    print("\nMultiple comparisons correction: Bonferroni")
    print("Family-wise error rate control: α_fw = 0.05")
    print("Per-comparison α = 0.05 / k, where k = number of comparisons\n")

    # Compare Probability-Based (best) vs others
    prob_scores = results["Probability"]["scores"]
    k = len(strategies) - 1  # number of comparisons vs Probability-Based
    alpha_per_comparison = 0.05 / k

    print(f"Number of comparisons (k): {k}")
    print(f"Per-comparison α (Bonferroni): {alpha_per_comparison:.6f}\n")

    comparisons = []
    for name, res in results.items():
        if name == "Probability":
            continue
        scores = res["scores"]
        t_stat, p_val = stats.ttest_ind(prob_scores, scores, equal_var=False)
        d = cohen_d(prob_scores, scores)
        glass = glass_delta(prob_scores, scores)
        hedges = hedges_g(prob_scores, scores)
        p_bonf = min(p_val * k, 1.0)

        comparisons.append({
            'name': name,
            'comparison': f"Probability vs {name}",
            't_stat': t_stat,
            'p_val': p_val,
            'p_bonf': p_bonf,
            'cohen_d': d,
            'glass_delta': glass,
            'hedges_g': hedges,
            'significant': p_bonf < alpha_per_comparison  # Use adjusted alpha, not 0.05
        })

        significance = "***" if p_bonf < 0.001 else "**" if p_bonf < 0.01 else "*" if p_bonf < 0.05 else "ns"
        print(f"Probability vs {name:15s}")
        print(f"  t-statistic      = {t_stat:8.3f}")
        print(f"  p-value (raw)    = {p_val:8.6f}")
        print(f"  p-value (Bonf.)  = {p_bonf:8.6f} {significance}")
        print(f"  Cohen's d        = {d:8.3f}")
        print(f"  Glass's Δ        = {glass:8.3f}")
        print(f"  Hedges' g        = {hedges:8.3f}")
        sig_result = p_bonf < alpha_per_comparison
        print(f"  Significant? (α={alpha_per_comparison:.4f}) = {'Yes' if sig_result else 'No'}")
        print()

    print("=" * 80)
    print("EFFECT SIZE INTERPRETATION")
    print("=" * 80)
    print("\nCohen's d guidelines:")
    print("  |d| < 0.2  : Negligible")
    print("  0.2 ≤ |d| < 0.5: Small")
    print("  0.5 ≤ |d| < 0.8: Medium")
    print("  |d| ≥ 0.8  : Large")
    print()

    # Save results
    with open('enhanced_statistical_results.txt', 'w') as f:
        f.write("ENHANCED STATISTICAL ANALYSIS RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"BOOTSTRAP CONFIDENCE INTERVALS ({BOOTSTRAP_SAMPLES} samples)\n")
        f.write("-" * 80 + "\n")
        for name, res in results.items():
            boot_low, boot_high = bootstrap_ci(res["scores"], n_bootstrap=BOOTSTRAP_SAMPLES)
            f.write(f"{name}: [{boot_low:.2f}, {boot_high:.2f}]\n")
        
        f.write("\n\nHYPOTHESIS TESTS WITH BONFERRONI CORRECTION\n")
        f.write("-" * 80 + "\n")
        for comp in comparisons:
            f.write(f"EV vs {comp['name']}:\n")
            f.write(f"  p-value (raw): {comp['p_val']:.6f}\n")
            f.write(f"  p-value (Bonf.): {comp['p_bonf']:.6f}\n")
            f.write(f"  Cohen's d: {comp['cohen_d']:.3f}\n")
            f.write(f"  Glass's Δ: {comp['glass_delta']:.3f}\n")
            f.write(f"  Hedges' g: {comp['hedges_g']:.3f}\n")
            f.write(f"  Significant: {comp['significant']}\n\n")

    print("Results saved to enhanced_statistical_results.txt")


if __name__ == "__main__":
    main()

