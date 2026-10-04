#!/usr/bin/env python3
"""
Generate Yahtzee visualization figures using yahtzee_simulator.

Usage:
    python generate_yahtzee_visualizations.py --games 10000
"""

import argparse
from yahtzee_simulator import (
    build_strategies,
    simulate_games,
    plot_mean_scores_with_ci,
    plot_score_boxplots,
    plot_high_score_probabilities,
    plot_score_density,
)


def main():
    parser = argparse.ArgumentParser(description="Generate Yahtzee visualizations")
    parser.add_argument("--games", type=int, default=10000,
                        help="Number of games per strategy")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    strategies = build_strategies()
    results = {}
    for name, strat in strategies.items():
        print(f"Simulating {name} for visualization with {args.games} games...")
        stats = simulate_games(strat, n_games=args.games, seed=args.seed)
        results[name] = stats

    print("Generating figures...")
    plot_mean_scores_with_ci(results, "yahtzee_mean_scores_ci.png")
    plot_score_boxplots(results, "yahtzee_score_boxplots.png")
    plot_high_score_probabilities(results, "yahtzee_high_score_probs.png")
    plot_score_density(results, "yahtzee_score_density.png")
    print("Figures saved:")
   

if __name__ == "__main__":
    main()
