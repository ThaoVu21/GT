Academic project — ISYE 6644: Simulation

## Project Overview
This project implements a Monte Carlo simulation to evaluate six different Yahtzee playing strategies and compare their performance through statistical analysis. The codebase has been improved with shared helper functions, static scoring methods, full Joker rule implementation, and enhanced expected value calculations.

## Files Included

### Main Code
- **yahtzee_simulator.py** (841 lines): The main simulation engine containing:
  - **Shared Helper Functions**: `is_potential_straight()`, `find_longest_run()`, `is_yahtzee()` - eliminate code duplication
  - **YahtzeeScoring Class**: Static scoring methods (no instance needed)
    - `score()`: Standard Yahtzee scoring
    - `handle_joker()`: Full Joker rule implementation
  - **Scorecard Class**: Tracks scores across 13 categories with bonus calculation
  - **Six Strategy Implementations**:
    1. `GreedyStrategy` - Baseline: maximizes immediate score
    2. `UpperSectionStrategy` - Focuses on upper section bonus (63 points)
    3. `ProbabilityStrategy` - Probability-based decisions with Yahtzee probability estimation
    4. `ExpectedValueStrategy` - Enhanced EV with approximate EV calculations (2000 simulations)
    5. `AdaptiveStrategy` - Adapts strategy based on game phase (early/mid/late)
    6. `YahtzeeFirstStrategy` - Aggressively pursues Yahtzee
  - **Simulation Functions**:
    - `play_single_game()`: Plays a single complete game (13 rounds, terminating simulation)
    - `simulate_games()`: Runs Monte Carlo simulations with comprehensive statistics
    - Returns: mean, median, std, SE, CI (95%), min, max, probabilities (P≥200, P≥225, P≥250, P≥275), timing
  - **Built-in Plotting Functions**:
    - `plot_mean_scores_with_ci()`: Bar chart with 95% CI error bars
    - `plot_score_boxplots()`: Box plots comparing distributions
    - `plot_high_score_probabilities()`: Line plot of high-score probabilities
    - `plot_score_density()`: Density plots using KDE (Kernel Density Estimation)
    - `plot_pvalue_heatmap()`: Heatmap of pairwise p-values
  - **CLI Interface**: Command-line arguments for games, strategies, seed, and plot generation
  - **Strategy Builder**: `build_strategies()` returns dictionary of all strategies

### Statistical Analysis
- **add_statistical_tests.py** (182 lines): Comprehensive statistical analysis including:
  - **Bootstrap Confidence Intervals**: 2000 bootstrap samples (optimized for performance)
  - **Normal Approximation Confidence Intervals**: Standard error-based CIs
  - **Hypothesis Testing**: Welch's t-test (unequal variances) for pairwise comparisons
  - **Bonferroni Correction**: Multiple comparisons correction (α = 0.05 / k comparisons)
  - **Effect Size Calculations**:
    - Cohen's d (pooled standard deviation)
    - Glass's Δ (uses control group std dev)
    - Hedges' g (bias-corrected Cohen's d)
  - **Output**: Console output and saves results to `enhanced_statistical_results.txt`
  - Compares "Probability" strategy (best performer) vs all other strategies

### Visualizations
- **generate_yahtzee_visualizations.py** (48 lines): Generates visualization figures using functions from `yahtzee_simulator.py`:
  - **yahtzee_mean_scores_ci.png**: Mean score comparison with 95% confidence interval error bars (Figure 1)
  - **yahtzee_score_boxplots.png**: Box plot comparison of score distributions (Figure 2)
  - **yahtzee_high_score_probs.png**: Probability of achieving high score thresholds (≥200, ≥225, ≥250, ≥275) (Figure 3)
  - **yahtzee_score_density.png**: Density plots comparing score distributions using KDE (Figure 4)
  - Uses `build_strategies()`, `simulate_games()`, and plot functions from the main simulator
  - CLI interface with `--games` and `--seed` arguments

  
  ## How to Run

### 1. Install Required Packages
```bash
pip install numpy matplotlib seaborn scipy weasyprint python-docx markdown
```

### 2. Run Simulations (Main Simulator)
```bash
# Run all strategies with default settings (10,000 games, seed=12345)
python3 yahtzee_simulator.py

# Custom number of games
python3 yahtzee_simulator.py --games 5000

# Run specific strategies only
python3 yahtzee_simulator.py --strategies "Probability,EV,Greedy"

# Disable plot generation
python3 yahtzee_simulator.py --no-plots

# Custom seed
python3 yahtzee_simulator.py --seed 42
```

Or use programmatically:
```python
from yahtzee_simulator import build_strategies, simulate_games
import random

random.seed(12345)
strategies = build_strategies()
results = {}

for name, strategy in strategies.items():
    stats = simulate_games(strategy, n_games=10000, seed=12345)
    results[name] = stats
    print(f"{name}: mean={stats['mean']:.2f}, CI=[{stats['ci_low']:.2f}, {stats['ci_high']:.2f}]")
```

### 3. Generate Visualizations
```bash
# Generate all 4 figures with default settings
python3 generate_yahtzee_visualizations.py

# Custom number of games
python3 generate_yahtzee_visualizations.py --games 10000

# Custom seed
python3 generate_yahtzee_visualizations.py --seed 12345
```

This will:
- Run 10,000 simulations for each of the 6 strategies (with specified seed)
- Generate 4 visualization files (PNG format, 300 DPI)
- Save figures to the current directory

### 4. Run Statistical Analysis
```bash
python3 add_statistical_tests.py
```

This will:
- Run 10,000 simulations for each strategy (seed=12345)
- Calculate bootstrap confidence intervals (2000 bootstrap samples)
- Perform hypothesis tests (Welch's t-test) with Bonferroni correction
- Compute effect sizes (Cohen's d, Glass's Δ, Hedges' g)
- Print comprehensive statistical summaries to console
- Save results to `enhanced_statistical_results.txt`

**Note**: The statistical analysis compares "Probability" strategy (best performer) vs all other strategies.


## Simulation Parameters
- **Default sample size**: 10,000 games per strategy
- **Default random seed**: 12345 (for reproducibility)
- **Number of strategies**: 6
- **Bootstrap iterations**: 2000 (optimized for performance in `add_statistical_tests.py`)
- **Significance level**: α = 0.05 (adjusted per comparison with Bonferroni correction)
- **Expected Value simulations**: 2000 simulations per decision (in `ExpectedValueStrategy`)

## Strategy Names
The strategies are identified by these names in the code:
- `"Greedy"` - GreedyStrategy
- `"UpperSection"` - UpperSectionStrategy
- `"Probability"` - ProbabilityStrategy
- `"EV"` - ExpectedValueStrategy
- `"Adaptive"` - AdaptiveStrategy
- `"YahtzeeFirst"` - YahtzeeFirstStrategy

## Dependencies
- Python 3.9+
- numpy
- matplotlib
- seaborn (for heatmap visualization)
- scipy (for statistical tests and KDE)
- markdown (for markdown parsing)

## Code Improvements
The simulator includes several improvements:
- **Shared Helper Functions**: Eliminates code duplication across strategies
- **Static Scoring Methods**: No repeated `YahtzeeGame` instantiation
- **Full Joker Rule**: Complete implementation of Yahtzee Joker rule
- **Enhanced Expected Value**: Approximate EV calculations with 2000 simulations
- **Standardized Category Priority**: Consistent category selection across strategies
- **Comprehensive Statistics**: Returns CI, SE, probabilities, and timing information
- **Built-in Plotting**: Plotting functions included in main simulator

## Results Summary
The Probability-Based strategy achieved the highest mean score of 171.21 points, outperforming the baseline Greedy strategy by 17.9%. Key findings:

- **Top Performers**: Probability-Based (171.21) and Expected Value (170.00) strategies significantly outperform heuristic approaches
- **Statistical Significance**: After Bonferroni correction, Probability-Based vs Expected Value is not statistically significant (p = 0.075 > 0.01), indicating they perform similarly
- **Effect Sizes**: Large effect sizes (Cohen's d ≈ 0.72) for Probability-Based vs Greedy/Upper Section Focus strategies
- **Performance Gap**: Best-performing strategies achieve approximately 67% of theoretical optimal play (254 points)


## Reproducibility
All simulations use a fixed random seed (12345 by default) for reproducibility. To reproduce results:
1. Use the same random seed (default: 12345)
2. Use the same sample size (default: 10,000 games per strategy)
3. Run all scripts in the order specified above

## Notes
- The statistical analysis includes proper multiple comparisons correction (Bonferroni)
- All confidence intervals are calculated using both bootstrap (2000 samples) and normal approximation methods
- Effect sizes are computed using three different measures for comprehensive comparison
- Visualizations are generated at 300 DPI for publication quality
- The simulator includes built-in plotting functions that can be called directly
- Strategy names in code differ slightly from display names (e.g., "Probability" vs "Probability-Based")
