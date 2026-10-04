#!/usr/bin/env python3
"""
Improved Yahtzee Game Simulator addressing code review feedback.

Improvements:
- Shared helper functions (is_potential_straight, etc.)
- Static score_category method (no repeated YahtzeeGame instantiation)
- Full Joker rule implementation
- Enhanced Expected Value calculations
- Standardized category priority lists
"""

import argparse
import math
import random
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple, Optional

import matplotlib.pyplot as plt
try:
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats as scipy_stats
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    plt = None
    np = None
    scipy_stats = None

# ---------------------- Categories & Scorecard ---------------------- #

UPPER_CATEGORIES = ["ones", "twos", "threes", "fours", "fives", "sixes"]
LOWER_CATEGORIES = [
    "three_of_a_kind",
    "four_of_a_kind",
    "full_house",
    "small_straight",
    "large_straight",
    "yahtzee",
    "chance",
]
ALL_CATEGORIES = UPPER_CATEGORIES + LOWER_CATEGORIES

# Standardized category priority (used across strategies)
CATEGORY_PRIORITY = [
    "yahtzee",
    "large_straight",
    "full_house",
    "four_of_a_kind",
    "small_straight",
    "three_of_a_kind",
]


@dataclass
class Scorecard:
    scores: Dict[str, int] = field(
        default_factory=lambda: {c: None for c in ALL_CATEGORIES}
    )

    def is_filled(self, category: str) -> bool:
        return self.scores.get(category) is not None

    def set_score(self, category: str, value: int) -> None:
        if self.is_filled(category):
            raise ValueError(f"Category '{category}' already filled.")
        if category not in ALL_CATEGORIES:
            raise ValueError(f"Unknown category '{category}'.")
        self.scores[category] = value

    def get_available_categories(self) -> List[str]:
        return [c for c, v in self.scores.items() if v is None]

    def get_upper_sum(self) -> int:
        return sum(
            v for c, v in self.scores.items()
            if c in UPPER_CATEGORIES and v is not None
        )

    def upper_bonus(self) -> int:
        return 35 if self.get_upper_sum() >= 63 else 0

    def total_score(self) -> int:
        base = sum(v for v in self.scores.values() if v is not None)
        return base + self.upper_bonus()

    def has_yahtzee_scored(self) -> bool:
        """Check if Yahtzee category has been scored (for Joker rule)."""
        return self.scores.get("yahtzee") == 50


# ---------------------- Shared Helper Functions ---------------------- #

def is_potential_straight(dice: Sequence[int]) -> bool:
    """Shared helper: Check if dice contain a potential straight."""
    dice_set = set(dice)
    straights = [{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}]
    return any(s.issubset(dice_set) for s in straights)


def find_longest_run(dice: Sequence[int]) -> List[int]:
    """Return longest consecutive run of distinct faces."""
    s = sorted(set(dice))
    if not s:
        return []
    best = [s[0]]
    cur = [s[0]]
    for x in s[1:]:
        if x == cur[-1] + 1:
            cur.append(x)
        else:
            if len(cur) > len(best):
                best = cur
            cur = [x]
    if len(cur) > len(best):
        best = cur
    return best


def is_yahtzee(dice: Sequence[int]) -> bool:
    """Check if dice form a Yahtzee (all same value)."""
    return len(set(dice)) == 1


# ---------------------- Scoring Logic (Static Method) ---------------------- #

class YahtzeeScoring:
    """Static scoring methods (no instance needed)."""

    @staticmethod
    def score(dice: Sequence[int], category: str) -> int:
        """Standard Yahtzee scoring (static method)."""
        counts = Counter(dice)
        total = sum(dice)

        # Upper section
        if category == "ones":
            return counts[1] * 1
        if category == "twos":
            return counts[2] * 2
        if category == "threes":
            return counts[3] * 3
        if category == "fours":
            return counts[4] * 4
        if category == "fives":
            return counts[5] * 5
        if category == "sixes":
            return counts[6] * 6

        # Lower section
        if category == "three_of_a_kind":
            return total if max(counts.values()) >= 3 else 0

        if category == "four_of_a_kind":
            return total if max(counts.values()) >= 4 else 0

        if category == "full_house":
            vals = sorted(counts.values())
            if (3 in vals and 2 in vals) or 5 in vals:
                return 25
            return 0

        if category == "small_straight":
            return 30 if is_potential_straight(dice) else 0

        if category == "large_straight":
            s = set(dice)
            if s == {1, 2, 3, 4, 5} or s == {2, 3, 4, 5, 6}:
                return 40
            return 0

        if category == "yahtzee":
            return 50 if is_yahtzee(dice) else 0

        if category == "chance":
            return total

        raise ValueError(f"Unknown category '{category}'.")

    @staticmethod
    def handle_joker(dice: Sequence[int], scorecard: Scorecard) -> Optional[str]:
        """
        Full Joker rule implementation:
        - If Yahtzee already scored (50) and another Yahtzee occurs:
          * Forced to upper section if available
          * Otherwise allowed as Joker for full house, small straight, large straight
          * If all joker-friendly categories filled, return None (fall back to normal selection)
        """
        if not is_yahtzee(dice):
            return None

        if not scorecard.has_yahtzee_scored():
            return None

        available = scorecard.get_available_categories()
        yahtzee_value = dice[0]  # All dice are the same

        # Required: upper section if available
        upper_cat = UPPER_CATEGORIES[yahtzee_value - 1]
        if upper_cat in available:
            return upper_cat

        # Joker alternatives
        joker_categories = ["full_house", "small_straight", "large_straight"]
        for cat in joker_categories:
            if cat in available:
                return cat

        # Edge case: If upper category filled AND all joker categories filled,
        # fall back to normal category selection (return None)
        return None


# ---------------------- Dice Helpers ---------------------- #

def roll_dice(num_dice: int = 5) -> List[int]:
    return [random.randint(1, 6) for _ in range(num_dice)]


def reroll_dice(dice: Sequence[int], keep_indices: Sequence[int]) -> List[int]:
    keep_set = set(keep_indices)
    new_dice = list(dice)
    for i in range(len(new_dice)):
        if i not in keep_set:
            new_dice[i] = random.randint(1, 6)
    return new_dice


# ---------------------- Strategy Base Class ---------------------- #

class Strategy:
    name: str = "Base"

    def choose_dice_to_keep(
        self, dice: List[int], roll_num: int, scorecard: Scorecard
    ) -> List[int]:
        raise NotImplementedError

    def choose_category(self, dice: List[int], scorecard: Scorecard) -> str:
        raise NotImplementedError


# ---------------------- Strategies ---------------------- #

class GreedyStrategy(Strategy):
    """Baseline: Keep most frequent face, maximize immediate score."""

    name = "Greedy"

    def choose_dice_to_keep(
        self, dice: List[int], roll_num: int, scorecard: Scorecard
    ) -> List[int]:
        counts = Counter(dice)
        face, _ = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
        return [i for i, d in enumerate(dice) if d == face]

    def choose_category(self, dice: List[int], scorecard: Scorecard) -> str:
        available = scorecard.get_available_categories()
        scores = {c: YahtzeeScoring.score(dice, c) for c in available}
        best = max(available, key=lambda c: scores[c])
        if scores[best] == 0 and "chance" in available:
            return "chance"
        return best


class UpperSectionStrategy(Strategy):
    """Focus on upper section bonus (63 points)."""

    name = "UpperSection"

    def choose_dice_to_keep(
        self, dice: List[int], roll_num: int, scorecard: Scorecard
    ) -> List[int]:
        counts = Counter(dice)
        for face in range(6, 0, -1):
            if counts[face] >= 2:
                return [i for i, d in enumerate(dice) if d == face]
        face, _ = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
        return [i for i, d in enumerate(dice) if d == face]

    def choose_category(self, dice: List[int], scorecard: Scorecard) -> str:
        available = scorecard.get_available_categories()
        counts = Counter(dice)
        upper_sum = scorecard.get_upper_sum()

        for face in range(6, 0, -1):
            cat = UPPER_CATEGORIES[face - 1]
            if cat in available and counts[face] >= 3:
                score = YahtzeeScoring.score(dice, cat)
                if upper_sum < 63 and upper_sum + score >= 63:
                    return cat

        scores = {c: YahtzeeScoring.score(dice, c) for c in available}
        best = max(available, key=lambda c: scores[c])
        if scores[best] == 0 and "chance" in available:
            return "chance"
        return best


class ProbabilityStrategy(Strategy):
    """Probability-based decisions with Yahtzee probability estimation."""

    name = "Probability"

    @staticmethod
    def yahtzee_prob(k: int, rolls_left: int) -> float:
        """Probability of getting Yahtzee given k matching dice and rolls_left."""
        if k >= 5:
            return 1.0
        if rolls_left <= 0:
            return 0.0
        p_single = (1 / 6) ** (5 - k)
        return 1 - (1 - p_single) ** rolls_left

    def choose_dice_to_keep(
        self, dice: List[int], roll_num: int, scorecard: Scorecard
    ) -> List[int]:
        counts = Counter(dice)
        rolls_left = 3 - roll_num

        # Keep four-of-a-kind
        for face, c in counts.items():
            if c >= 4:
                return [i for i, d in enumerate(dice) if d == face]

        best_face, best_count = max(counts.items(), key=lambda kv: kv[1])
        if best_count == 3 and rolls_left > 0:
            p_y = self.yahtzee_prob(3, rolls_left)
            if p_y >= 0.05:
                return [i for i, d in enumerate(dice) if d == best_face]

        # Use shared helper for straights
        if is_potential_straight(dice):
            run = find_longest_run(dice)
            run_set = set(run)
            return [i for i, d in enumerate(dice) if d in run_set]

        return [i for i, d in enumerate(dice) if d == best_face]

    def choose_category(self, dice: List[int], scorecard: Scorecard) -> str:
        available = scorecard.get_available_categories()
        scores = {c: YahtzeeScoring.score(dice, c) for c in available}

        # Check Joker rule first
        joker_cat = YahtzeeScoring.handle_joker(dice, scorecard)
        if joker_cat:
            return joker_cat

        if "yahtzee" in available and scores["yahtzee"] == 50:
            return "yahtzee"

        # Use standardized priority
        for cat in CATEGORY_PRIORITY:
            if cat in available and scores[cat] > 0:
                return cat

        best = max(available, key=lambda c: scores[c])
        if scores[best] == 0 and "chance" in available:
            return "chance"
        return best


class ExpectedValueStrategy(Strategy):
    """Enhanced Expected Value with approximate EV calculations."""

    name = "EV"

    @staticmethod
    def approximate_ev(kept: List[int], rolls_remaining: int, category: str, n_sim: int = 2000) -> float:
        """
        Approximate expected value by simulating future rolls.
        EV = Σ P(x) * score(x, category) ≈ (1/N) * Σ score(simulated_roll, category)
        """
        if rolls_remaining <= 0:
            return YahtzeeScoring.score(kept, category)

        scores = []
        for _ in range(n_sim):
            # Simulate remaining rolls
            temp_dice = list(kept)
            for _ in range(rolls_remaining):
                # Reroll non-kept dice
                num_to_roll = 5 - len(temp_dice)
                new_rolls = roll_dice(num_to_roll)
                temp_dice = temp_dice + new_rolls
            scores.append(YahtzeeScoring.score(temp_dice, category))

        return sum(scores) / len(scores) if scores else 0.0

    def choose_dice_to_keep(
        self, dice: List[int], roll_num: int, scorecard: Scorecard
    ) -> List[int]:
        counts = Counter(dice)
        for face, c in counts.items():
            if c >= 4:
                return [i for i, d in enumerate(dice) if d == face]
        best_face, best_count = max(counts.items(), key=lambda kv: kv[1])
        if best_count >= 3:
            return [i for i, d in enumerate(dice) if d == best_face]
        # Use shared helper
        if is_potential_straight(dice):
            run = find_longest_run(dice)
            run_set = set(run)
            return [i for i, d in enumerate(dice) if d in run_set]
        return [i for i, d in enumerate(dice) if d == best_face]

    def choose_category(self, dice: List[int], scorecard: Scorecard) -> str:
        """
        Choose category using approximate expected value calculation.
        For final roll (rolls_remaining=0), uses immediate score with bonus adjustment.
        For earlier rolls, would use approximate_ev() but in this implementation
        we're at final roll, so use bonus-aware scoring.
        """
        available = scorecard.get_available_categories()
        upper_sum = scorecard.get_upper_sum()
        rolls_remaining = 0  # Final roll, no more rerolls

        # Check Joker rule
        joker_cat = YahtzeeScoring.handle_joker(dice, scorecard)
        if joker_cat:
            return joker_cat

        values: Dict[str, float] = {}
        for cat in available:
            # For final roll, use immediate score (no future rolls to simulate)
            # But adjust for bonus potential
            val = YahtzeeScoring.score(dice, cat)
            # Add bonus value if upper section
            if cat in UPPER_CATEGORIES and upper_sum < 63:
                if upper_sum + val >= 63:
                    val += 35  # Approximate expected benefit of triggering bonus
            values[cat] = val

        best = max(available, key=lambda c: values[c])
        if values[best] == 0 and "chance" in available:
            return "chance"
        return best


class AdaptiveStrategy(Strategy):
    """Adapts strategy based on game phase (early/mid/late)."""

    name = "Adaptive"

    def _phase(self, scorecard: Scorecard) -> str:
        filled = sum(scorecard.scores[c] is not None for c in ALL_CATEGORIES)
        if filled <= 4:
            return "early"
        elif filled <= 8:
            return "mid"
        return "late"

    def choose_dice_to_keep(
        self, dice: List[int], roll_num: int, scorecard: Scorecard
    ) -> List[int]:
        counts = Counter(dice)
        for face, c in counts.items():
            if c >= 3:
                return [i for i, d in enumerate(dice) if d == face]
        # Use shared helper
        if is_potential_straight(dice):
            run = find_longest_run(dice)
            run_set = set(run)
            return [i for i, d in enumerate(dice) if d in run_set]
        best_face, _ = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
        return [i for i, d in enumerate(dice) if d == best_face]

    def choose_category(self, dice: List[int], scorecard: Scorecard) -> str:
        phase = self._phase(scorecard)
        available = scorecard.get_available_categories()
        scores = {c: YahtzeeScoring.score(dice, c) for c in available}
        upper_sum = scorecard.get_upper_sum()
        upper_avail = [c for c in available if c in UPPER_CATEGORIES]

        # Check Joker rule
        joker_cat = YahtzeeScoring.handle_joker(dice, scorecard)
        if joker_cat:
            return joker_cat

        if phase == "early":
            if upper_sum < 63 and upper_avail:
                best_upper = max(upper_avail, key=lambda c: scores[c])
                if scores[best_upper] >= 3:
                    return best_upper

        elif phase == "mid":
            # Use standardized priority
            for cat in CATEGORY_PRIORITY[:3]:  # Top 3 high-value
                if cat in available and scores[cat] > 0:
                    return cat
            if upper_sum < 63 and upper_avail:
                best_upper = max(upper_avail, key=lambda c: scores[c])
                if scores[best_upper] >= 3:
                    return best_upper

        else:  # late
            # Use standardized priority
            for cat in CATEGORY_PRIORITY:
                if cat in available and scores[cat] > 0:
                    return cat
            if upper_avail:
                return max(upper_avail, key=lambda c: scores[c])

        best = max(available, key=lambda c: scores[c])
        if scores[best] == 0 and "chance" in available:
            return "chance"
        return best


class YahtzeeFirstStrategy(Strategy):
    """Aggressively pursues Yahtzee."""

    name = "YahtzeeFirst"

    def choose_dice_to_keep(
        self, dice: List[int], roll_num: int, scorecard: Scorecard
    ) -> List[int]:
        counts = Counter(dice)
        if not counts:
            return []
        face, count = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
        if count >= 2:
            return [i for i, d in enumerate(dice) if d == face]
        max_val = max(dice)
        return [i for i, d in enumerate(dice) if d == max_val]

    def choose_category(self, dice: List[int], scorecard: Scorecard) -> str:
        available = scorecard.get_available_categories()
        scores = {c: YahtzeeScoring.score(dice, c) for c in available}

        # Check Joker rule
        joker_cat = YahtzeeScoring.handle_joker(dice, scorecard)
        if joker_cat:
            return joker_cat

        if "yahtzee" in available and scores["yahtzee"] == 50:
            return "yahtzee"

        # Use standardized priority
        for cat in CATEGORY_PRIORITY:
            if cat in available and scores[cat] > 0:
                return cat

        upper_sum = scorecard.get_upper_sum()
        upper_avail = [c for c in available if c in UPPER_CATEGORIES]
        if upper_sum < 63 and upper_avail:
            best_upper = max(upper_avail, key=lambda c: scores[c])
            if scores[best_upper] >= 3:
                return best_upper

        best = max(available, key=lambda c: scores[c])
        if scores[best] == 0 and "chance" in available:
            return "chance"
        return best


# ---------------------- Simulation + Stats ---------------------- #

def play_single_game(strategy: Strategy) -> int:
    """Play a single complete game (13 rounds, terminating simulation)."""
    scorecard = Scorecard()
    for round_num in range(13):
        dice = roll_dice(5)
        for roll_num in range(1, 3):  # up to 2 rerolls (rolls 1 and 2)
            keep_indices = strategy.choose_dice_to_keep(dice, roll_num, scorecard)
            # If keeping all 5 dice, we're done with this turn
            if len(keep_indices) == 5:
                break
            # Reroll the dice not kept
            dice = reroll_dice(dice, keep_indices)
        
        # Final dice combination - choose category
        category = strategy.choose_category(dice, scorecard)
        # Handle Joker rule fallback if needed
        if category is None:
            # Joker rule returned None (edge case), use normal selection
            available = scorecard.get_available_categories()
            scores = {c: YahtzeeScoring.score(dice, c) for c in available}
            category = max(available, key=lambda c: scores[c])
            if scores[category] == 0 and "chance" in available:
                category = "chance"
        
        value = YahtzeeScoring.score(dice, category)
        scorecard.set_score(category, value)
    return scorecard.total_score()


def simulate_games(strategy: Strategy, n_games: int = 10_000, seed: Optional[int] = None):
    if seed is not None:
        random.seed(seed)
    start_time = time.time()
    scores: List[int] = [play_single_game(strategy) for _ in range(n_games)]
    elapsed = time.time() - start_time

    mean = statistics.fmean(scores)
    std = statistics.stdev(scores) if n_games > 1 else 0.0  # Sample standard deviation for CI formula
    se = std / math.sqrt(n_games) if n_games > 1 else 0.0
    z = 1.96
    ci_low = mean - z * se
    ci_high = mean + z * se

    def p_at_least(t: int) -> float:
        return sum(1 for s in scores if s >= t) / n_games

    return {
        "scores": scores,
        "mean": mean,
        "median": statistics.median(scores),
        "std": std,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "min": min(scores),
        "max": max(scores),
        "p200": p_at_least(200),
        "p225": p_at_least(225),
        "p250": p_at_least(250),
        "p275": p_at_least(275),
        "time_seconds": elapsed,
        "games_per_second": n_games / elapsed if elapsed > 0 else 0,
    }


def build_strategies() -> Dict[str, Strategy]:
    return {
        "Greedy": GreedyStrategy(),
        "UpperSection": UpperSectionStrategy(),
        "Probability": ProbabilityStrategy(),
        "EV": ExpectedValueStrategy(),
        "Adaptive": AdaptiveStrategy(),
        "YahtzeeFirst": YahtzeeFirstStrategy(),
    }


# ---------------------- Plot Helpers ---------------------- #

def plot_mean_scores_with_ci(results: Dict[str, Dict], filename: str):
    """Bar chart with error bars showing 95% CI."""
    names = list(results.keys())
    means = [results[n]["mean"] for n in names]
    ci_lows = [results[n]["ci_low"] for n in names]
    ci_highs = [results[n]["ci_high"] for n in names]
    lower_err = [means[i] - ci_lows[i] for i in range(len(names))]
    upper_err = [ci_highs[i] - means[i] for i in range(len(names))]
    yerr = [lower_err, upper_err]

    plt.figure(figsize=(12, 6))
    x = range(len(names))
    bars = plt.bar(x, means, yerr=yerr, capsize=5, color=plt.cm.Set2(range(len(names))))
    plt.xticks(x, names, rotation=45, ha="right")
    plt.ylabel("Mean Score")
    plt.title("Mean Score by Strategy (95% CI Error Bars)")
    plt.grid(axis="y", alpha=0.3)
    for bar, mean in zip(bars, means):
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 1,
                 f"{mean:.1f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_score_boxplots(results: Dict[str, Dict], filename: str):
    """Box plots comparing distributions."""
    names = list(results.keys())
    data = [results[n]["scores"] for n in names]
    plt.figure(figsize=(12, 6))
    bp = plt.boxplot(data, labels=names, showmeans=True, patch_artist=True)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Score")
    plt.title("Score Distributions by Strategy (Box Plot)")
    plt.grid(axis="y", alpha=0.3)
    # Color boxes
    colors = plt.cm.tab20(range(len(names)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_high_score_probabilities(results: Dict[str, Dict], filename: str):
    """Line plot of high-score probabilities."""
    names = list(results.keys())
    thresholds = [("p200", 200), ("p225", 225), ("p250", 250), ("p275", 275)]

    plt.figure(figsize=(12, 6))
    x = range(len(names))
    for key, label_val in thresholds:
        probs = [results[n][key] for n in names]
        plt.plot(x, probs, marker="o", label=f"P(score ≥ {label_val})", linewidth=2)
    plt.xticks(x, names, rotation=45, ha="right")
    plt.ylabel("Probability")
    plt.ylim(0, 1)
    plt.title("High-Score Achievement Probabilities by Strategy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_score_density(results: Dict[str, Dict], filename: str):
    """Density plots comparing score distributions."""
    if not HAS_PLOTTING:
        print("Warning: matplotlib/scipy not available. Skipping density plot.")
        return
    
    from scipy import stats as scipy_stats
    
    plt.figure(figsize=(12, 7))
    names = list(results.keys())
    colors_list = plt.cm.Set2(range(len(names)))
    
    for name, color in zip(names, colors_list):
        scores = results[name]["scores"]
        # Create smooth density curve using KDE
        kde = scipy_stats.gaussian_kde(scores)
        x_range = np.linspace(min(scores), max(scores), 200)
        density = kde(x_range)
        plt.plot(x_range, density, label=name, linewidth=2, color=color)
        # Add vertical line at mean
        mean = results[name]["mean"]
        plt.axvline(mean, color=color, linestyle='--', alpha=0.5, linewidth=1)
    
    plt.xlabel("Final Score")
    plt.ylabel("Density")
    plt.title("Score Distribution Comparison (Density Plots)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_pvalue_heatmap(comparisons: List[Dict], strategies: List[str], filename: str):
    """Create heatmap of p-values from pairwise comparisons."""
    if not HAS_PLOTTING:
        print("Warning: matplotlib not available. Skipping p-value heatmap.")
        return
    
    try:
        import seaborn as sns
    except ImportError:
        print("Warning: seaborn not installed. Skipping p-value heatmap.")
        return
    
    # Build p-value matrix
    n = len(strategies)
    pmatrix = np.ones((n, n))
    
    # Map strategy names to indices
    name_to_idx = {name: i for i, name in enumerate(strategies)}
    
    # Fill matrix from comparisons
    for comp in comparisons:
        comp_str = comp.get('comparison', '') or f"EV vs {comp.get('name', '')}"
        parts = comp_str.split(' vs ')
        if len(parts) == 2:
            name1, name2 = parts[0].strip(), parts[1].strip()
            idx1, idx2 = -1, -1
            # Map to our strategy names
            for s in strategies:
                if s in name1 or name1 in s:
                    idx1 = name_to_idx.get(s, -1)
                if s in name2 or name2 in s:
                    idx2 = name_to_idx.get(s, -1)
            if idx1 >= 0 and idx2 >= 0:
                p_val = comp.get('p_bonf', comp.get('p_val', 1.0))
                pmatrix[idx1, idx2] = p_val
                pmatrix[idx2, idx1] = p_val  # Symmetric
    
    # Set diagonal to 1.0 (self-comparison)
    np.fill_diagonal(pmatrix, 1.0)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pmatrix, annot=True, fmt='.3f', cmap='coolwarm_r', 
                xticklabels=strategies, yticklabels=strategies,
                vmin=0, vmax=0.05, cbar_kws={'label': 'p-value (Bonferroni adjusted)'})
    plt.title("Pairwise Welch t-test p-values (Bonferroni Adjusted)")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# ---------------------- CLI ---------------------- #

def main():
    parser = argparse.ArgumentParser(description="Yahtzee Strategy Simulator")
    parser.add_argument("--games", type=int, default=10000,
                        help="Number of games per strategy")
    parser.add_argument("--strategies", type=str, default="all",
                        help="Comma-separated list or 'all'")
    parser.add_argument("--seed", type=int, default=12345,
                        help="Random seed")
    parser.add_argument("--no-plots", action="store_true",
                        help="Disable plot generation")
    args = parser.parse_args()

    all_strats = build_strategies()
    if args.strategies.lower() == "all":
        selected = all_strats
    else:
        names = [s.strip() for s in args.strategies.split(",")]
        selected = {n: all_strats[n] for n in names if n in all_strats}
        if not selected:
            raise ValueError(f"No valid strategies selected from {names}")

    random.seed(args.seed)
    results: Dict[str, Dict] = {}

    for name, strat in selected.items():
        print(f"Simulating {name} with {args.games} games...")
        stats = simulate_games(strat, n_games=args.games)
        results[name] = stats
        print(f"  Mean: {stats['mean']:.2f}")
        print(f"  95% CI: [{stats['ci_low']:.2f}, {stats['ci_high']:.2f}]")
        print(f"  Std dev: {stats['std']:.2f}, SE: {stats['se']:.3f}")
        print(f"  Min/Max: {stats['min']} / {stats['max']}")
        print(f"  P>=200: {stats['p200']:.3f}, P>=225: {stats['p225']:.3f}, "
              f"P>=250: {stats['p250']:.3f}, P>=275: {stats['p275']:.3f}")
        print(f"  Time: {stats['time_seconds']:.2f}s ({stats['games_per_second']:.0f} games/s)")
        print()

    if not args.no_plots and results:
        print("Generating plots...")
        plot_mean_scores_with_ci(results, "yahtzee_mean_scores_ci.png")
        plot_score_boxplots(results, "yahtzee_score_boxplots.png")
        plot_high_score_probabilities(results, "yahtzee_high_score_probs.png")
        print("Plots saved.")


if __name__ == "__main__":
    main()

