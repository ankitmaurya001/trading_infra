# Optimization Stability Roadmap

## The Problem

When running `automate_ma_walkforward_kite.py`, the validation PnL flips between
positive and negative depending on the `--start-date`. Shifting it by even a few
days can completely reverse the results.

This is not a bug. It is a well-documented statistical phenomenon called
**data-mining bias**, described in detail by David Aronson in *Evidence-Based
Technical Analysis* (Wiley, 2007). The root cause is that our optimizer searches
2,431 parameter combinations but evaluates them on very few actual trades, so it
picks parameters that happened to get lucky, not parameters that found a real
pattern. When those lucky parameters enter the validation window, their luck
doesn't carry over and PnL becomes random.

---

## How the Current System Works

```
For each iteration:
  1. OPTIMIZE: Take 30 days of 15m OHLC data
     - Test 11 short windows x 17 long windows x 13 RR ratios = 2,431 combos
     - Score each combo with composite_score (Sharpe, Calmar, profit factor, etc.)
     - Pick the top 5 by composite_score (one per risk-reward ratio)

  2. VALIDATE: Take the next 30 days of data
     - Run the 5 selected parameter sets through majority-vote validation
     - Record PnL, trades, win rate

  3. ADVANCE: Move the window forward and repeat
```

The problem lives in Step 1. With only 30 days of 15-minute data on MCX, most
parameter combinations produce only 2-6 trades during optimization. A Sharpe
ratio computed on 3 trades is meaningless. The optimizer is essentially picking
the combo that got lucky on a coin flip.

---

## The Problems (Ranked by Impact)

### Problem 1: Too Few Trades in the Optimization Window

**Severity: Critical**

With `--optimization-days 30` and 15-minute bars on MCX (~60 bars/day), the
optimizer gets roughly 1,800 price bars. But for MA crossovers with parameters
like short=50, long=150, this produces only 2-6 actual crossover trades. The
composite_score is computed from those 2-6 trades.

Aronson's experiments (Chapter 6, Figures 6.49-6.52) prove that when the number
of observations used to score rules is small, the data-mining bias becomes
enormous. With just 2 observations, the bias can exceed 200% per year. With
1,000 observations, the bias drops to under 3%.

**The effective sample size is the trade count, not the bar count.** This is the
single most important insight.

### Problem 2: No Statistical Significance Test

**Severity: Critical**

After the optimizer finds "the best" parameters, we proceed directly to
validation without asking: "Is this result statistically significant, or could
the optimizer have achieved the same score on purely random data?"

Aronson's central recommendation is the Monte Carlo Permutation Test (Chapter 6,
pp. 327-329). It answers this question by:

1. Recording the best composite_score from the real optimization
2. Shuffling the data to destroy any temporal patterns (permute returns)
3. Re-running the full optimization on the shuffled data
4. Repeating 100-500 times to build a distribution of "best scores on noise"
5. Checking if the real score beats 95% of the noise scores (p < 0.05)

If the optimizer's result is not statistically significant, skip that iteration
entirely. This directly prevents trading on noise.

### Problem 3: Composite Score Doesn't Account for Market Trend

**Severity: High**

If the market trends strongly upward during the optimization window, every
long-biased MA combination will score well regardless of its actual predictive
power. The composite_score doesn't subtract the buy-and-hold return from each
rule's return.

Aronson calls this "detrending" (Appendix, pp. 475-476) and proves it is
mathematically equivalent to benchmarking. Without it, the optimizer confuses
"the market went up" with "this rule is good."

### Problem 4: No Minimum Trade Filter

**Severity: High**

A parameter combination that happened to take 1 trade during optimization and
that 1 trade was a winner will have a perfect Sharpe ratio, perfect win rate,
zero drawdown, and a composite_score near 1.0. It will win the optimization
competition. But this result is pure noise.

There is no filter that says "ignore any parameter combination with fewer than
N trades." Without this, the optimizer gravitates toward sparse-trade
combinations that got lucky once.

### Problem 5: Parameter Selection Picks Isolated Peaks

**Severity: Medium**

The optimizer picks the single highest-scoring combination per risk-reward
ratio. It does not check whether nearby parameter combinations also score well.

If short=20, long=100 scores 0.85 but short=15, long=95 and short=25, long=105
both score 0.30, then 0.85 is an isolated spike -- likely noise. If all three
score 0.80+, that is a stable plateau -- likely a real pattern.

### Problem 6: No Walk-Forward Efficiency Gate

**Severity: Medium**

The system has no way to measure whether the optimization result carried forward
into validation. A simple ratio -- validation performance divided by
optimization performance -- would reveal iterations where the optimizer's
findings didn't generalize. These could be flagged or excluded from compounded
results.

---

## The Solutions (In Priority Order)

### Priority 1: Increase Optimization Window to 90 Days

**Effort: Trivial (change one default)**
**Impact: Very High**
**File: `automate_ma_walkforward_kite.py` line 55**

Change `DEFAULT_OPTIMIZATION_DAYS` from 30 to 90 (or 120).

With 90 days of MCX 15m data:
- ~5,400 price bars instead of ~1,800
- ~8-20 crossover trades per combination instead of 2-6
- Composite scores become statistically meaningful
- Data-mining bias drops from "extreme" to "moderate"

This is the single most effective change. Aronson's experiments show the
data-mining bias drops exponentially as sample size increases. Going from 30 to
90 days roughly triples the trade count, which dramatically reduces the chance
of a lucky combination winning the competition.

**Trade-off:** The validation window stays the same size, so you'll get fewer
walk-forward iterations over the same historical period.

**Alternative -- Anchored walk-forward:** Instead of a fixed 90-day rolling
window, always start optimization from the very first available date. Each
iteration uses a growing optimization window. Iteration 1 uses 90 days,
iteration 2 uses 120 days, iteration 3 uses 150 days, etc. This maximizes
sample size while still validating on unseen future data.

### Priority 2: Add Minimum Trade Filter

**Effort: Small**
**Impact: High**
**File: `run_ma_optimization_kite_parallel.py` (inside the worker function)**

Before computing composite_score for any parameter combination, check the
number of completed trades. If fewer than 10 trades, set composite_score to 0.

Why 10? It's a pragmatic threshold. With fewer than 10 trades:
- Win rate is measured in 10% increments (0%, 10%, 20%...) -- too coarse
- Sharpe ratio has huge confidence intervals
- A single lucky trade dominates all metrics
- The composite_score cannot distinguish signal from noise

With 10+ trades, the metrics start to have statistical meaning.

### Priority 3: Detrend Returns Before Scoring

**Effort: Medium**
**Impact: High**
**File: `ma_3d_optimization_visualizer.py` (composite_score computation)**

Before computing composite_score for each parameter combination:

1. Calculate the buy-and-hold return for the optimization period
2. Subtract this return from the rule's return
3. Use the excess return (not the raw return) in the composite_score formula

This ensures the optimizer selects parameters that beat the market, not
parameters that simply rode the market's direction. A rule that returned 15% in
a period where buy-and-hold returned 12% has only 3% of genuine alpha. Without
detrending, the optimizer treats the full 15% as the rule's merit.

### Priority 4: Monte Carlo Permutation Significance Test

**Effort: Medium**
**Impact: Very High**
**File: New module (e.g., `monte_carlo_significance.py`) + integration into `automate_ma_walkforward_kite.py`**

After the optimizer selects the Top N parameter sets but before validation, we
test each parameter set to determine whether it found a real pattern or just
got lucky on noise.

#### The Hypothesis Test

**Null Hypothesis (H0):** The parameter set's score is a result of random
chance. The MA crossover did not capture any genuine pattern in the price
action. Shuffling the time order of the data should produce similar scores.

**Alternative Hypothesis (Ha):** The parameter set's score is significantly
higher than what random noise produces. The MA parameters captured a real
trend or pattern that only exists in the actual time-ordered data.

#### The Core Idea

If a strategy found a real pattern, its score depends on the time ordering of
the data (trends require prices to move in a sustained direction over time).
Shuffling the data destroys that ordering. So a genuine strategy will score
well on real data but poorly on shuffled data. A lucky strategy will score
about the same on both -- it never needed real patterns to begin with.

#### How to Shuffle the Data (15-Minute Bar Returns)

We use the real OHLC data but destroy its time ordering by shuffling the
15-minute bar-to-bar returns:

```
Step 1: Compute close-to-close returns for each 15m bar

  Bar 1→2 return = (Close[2] - Close[1]) / Close[1]  →  +0.24%
  Bar 2→3 return = (Close[3] - Close[2]) / Close[2]  →  -0.57%
  Bar 3→4 return = (Close[4] - Close[3]) / Close[3]  →  +0.25%
  ...
  (1,799 returns from 1,800 bars)

Step 2: Record each bar's internal OHLC shape (ratios relative to Close)

  For bar i:
    o_ratio[i] = Open[i]  / Close[i]
    h_ratio[i] = High[i]  / Close[i]
    l_ratio[i] = Low[i]   / Close[i]

Step 3: Randomly shuffle the 1,799 returns

  Original order: [+0.24%, -0.57%, +0.25%, ...]
  Shuffled order:  [-0.12%, +0.83%, -0.57%, +0.24%, ...]

Step 4: Rebuild synthetic Close prices from the shuffled returns

  Synthetic Close[1] = 245.3  (same starting price)
  Synthetic Close[2] = 245.3 × (1 + shuffled_return[1])
  Synthetic Close[3] = Close[2] × (1 + shuffled_return[2])
  ...

Step 5: Rebuild O, H, L using the original bar shapes

  Synthetic Open[j]  = Synthetic Close[j] × o_ratio[j]
  Synthetic High[j]  = Synthetic Close[j] × h_ratio[j]
  Synthetic Low[j]   = Synthetic Close[j] × l_ratio[j]
```

The result: synthetic OHLC data that has the same volatility, same candle
shapes, and same return distribution as the real data -- but with all temporal
patterns (trends, momentum, mean-reversion) destroyed.

#### The Full Procedure

```
For each of the Top N parameter sets (e.g., short=20, long=100, RR=4.5):

  1. Run strategy on REAL data → compute mean return → real_score

  2. For i in 1..1000:
     a. Shuffle 15-minute returns (as described above)
     b. Rebuild synthetic OHLC price series
     c. Run the SAME strategy (same short, long, RR) on synthetic data
     d. Compute mean return → shuffled_scores[i]

  3. p_value = count(shuffled_scores >= real_score) / 1000

  4. Decision:
     If p_value < 0.01  → PASS (this parameter set is significant)
     If p_value >= 0.01 → FAIL (this parameter set was likely noise)
```

We use p < 0.01 (not 0.05) as the threshold because the Top N parameters were
already selected from 2,431 candidates. The stricter threshold compensates for
this selection bias (Bonferroni-style correction: 0.05 / TOP_N ≈ 0.01).

#### Example: Significant vs. Not Significant

**Parameter set that found a real trend:**
```
Real data mean return:      +2.8%
Shuffled data mean returns: [-0.5%, +0.3%, -1.1%, +0.6%, -0.2%, ...]
  (mostly near zero -- no trend to exploit on shuffled data)

Shuffled scores >= 2.8%:  3 out of 1000
p-value = 0.003 < 0.01 → PASS

Interpretation: Shuffling the data destroyed the strategy's edge. This means
the edge depended on the real time ordering (i.e., actual trends). The
strategy captured something genuine.
```

**Parameter set that got lucky on noise:**
```
Real data mean return:      +1.9%
Shuffled data mean returns: [+1.5%, +2.1%, +1.8%, +2.4%, +1.3%, ...]
  (similar scores even on shuffled data)

Shuffled scores >= 1.9%:  412 out of 1000
p-value = 0.412 > 0.01 → FAIL

Interpretation: Shuffling the data did NOT hurt the strategy's score. The
strategy scores just as well on random noise as on real data. It was not
capturing any genuine pattern -- it was just getting lucky.
```

#### Decision Rule After Testing All Top N

| Outcome | Action |
|---|---|
| 0 out of 5 pass | Skip iteration entirely (optimizer found only noise) |
| 1-2 out of 5 pass | Proceed to validation with only the survivors |
| 3-5 out of 5 pass | Proceed to validation with all survivors (strong signal) |

#### Why This Approach Is Practical

This tests each of the Top N parameter sets individually rather than re-running
the entire 2,431-combination optimization on each shuffle. Per iteration:

- Full Aronson approach: 2,431 combos × 1,000 shuffles = 2,431,000 backtests
- This approach: 5 param sets × 1,000 shuffles = 5,000 backtests

That is **486 times faster** while still catching the primary failure mode:
parameter sets that scored well on 2-3 lucky trades that won't repeat.

#### Which Metric to Use for the Score

Use **mean return** (not Sharpe, not composite_score):
- Simplest and fastest to compute
- No edge cases (Sharpe can be undefined when stdev is zero)
- This is what Aronson uses in his case study
- If mean return is significant, Sharpe and composite will follow

### Priority 5: Parameter Neighborhood Stability Check

**Effort: Medium**
**Impact: Medium**
**File: `run_ma_optimization_kite_parallel_top3.py` (selection logic)**

After identifying the top candidates by composite_score, verify that their
neighbors in parameter space also score well:

```
For each top-N candidate (short, long, rr):
  Collect scores of neighbors:
    (short +/- 1 step, long +/- 1 step, same rr)
  Compute average neighbor score

  If average neighbor score < 50th percentile of all scores:
    Reject this candidate (isolated peak = likely noise)
  Else:
    Accept (broad plateau = likely robust)
```

This is a form of regularization. A parameter set on a broad plateau in the
score surface is more likely to represent a genuine pattern, because slightly
different parameters also capture it. An isolated spike suggests the exact
parameter values happened to align with noise in the specific data window.

### Priority 6: Walk-Forward Efficiency Tracking

**Effort: Small**
**Impact: Medium**
**File: `automate_ma_walkforward_kite.py` (in `run_iteration` and summary)**

After each iteration completes, compute:

```
WFE = validation_mean_return / optimization_mean_return
```

Track this in `IterationSummary` and in the output CSV. Use it as a diagnostic:

- WFE > 0.5: Good. Optimization results generalized well.
- WFE 0.3-0.5: Acceptable. Some deterioration is normal.
- WFE < 0.3: Poor. The optimizer likely found noise.
- WFE < 0: Bad. Validation went the opposite direction.

Over time, this metric helps calibrate trust in the optimizer's output and
reveals which market conditions are favorable for the strategy.

---

## Implementation Sequence

```
Phase 1 (Quick Wins -- hours of work)
  [1] Increase optimization window to 90 days
  [2] Add minimum 10-trade filter in optimization grid
  [6] Add WFE tracking to iteration summary

Phase 2 (Medium Effort -- 1-2 days)
  [3] Detrend returns before composite_score calculation
  [4] Monte Carlo Permutation significance test
      - New module: monte_carlo_significance.py
      - Shuffle 15m bar returns, rebuild synthetic OHLC
      - Test each Top N param set against 1,000 shuffles
      - Use p < 0.01 threshold (Bonferroni-corrected)
      - Integrate as gate between optimization and validation
  [5] Add parameter neighborhood stability check

Phase 3 (Integration & Polish -- 1-2 days)
  - Wire Monte Carlo p-values into iteration summary CSV
  - Add skip/proceed logic based on significance results
  - Parallel execution of shuffled backtests across CPU cores
```

Phase 1 alone should noticeably reduce the start-date sensitivity. Phase 2
adds the statistical significance gate and additional robustness. Phase 3
completes the integration and optimizes the computational cost.

---

## How to Know If It's Working

After implementing changes, run the walk-forward with 3-4 different start dates
that differ by only a few days (e.g., Sept 1, Sept 3, Sept 5, Sept 7). Compare:

**Before improvements:** Validation PnL flips sign across start dates. One run
shows +5%, another shows -8%. The selected parameters are completely different
each time.

**After improvements:** Validation PnL is more consistent across start dates.
It may still vary in magnitude, but the sign should be stable. The selected
parameters should overlap significantly across runs. Iterations that the Monte
Carlo test flagged as insignificant should be the ones where PnL was random.

Zero variance across start dates is not the goal (some sensitivity is
inevitable in financial data). The goal is that the system stops trading on
noise and only acts when it finds statistically defensible signals.

---

## References

- Aronson, D. R. (2007). *Evidence-Based Technical Analysis: Applying the
  Scientific Method and Statistical Inference to Trading Signals*. Wiley.
  - Chapter 6: Data-Mining Bias (pp. 255-330) -- core theory
  - Chapter 6 Solutions section (pp. 320-330) -- Monte Carlo and Bootstrap
  - Appendix (pp. 475-476) -- detrending proof
- White, H. (2000). "A Reality Check for Data Snooping." *Econometrica*.
- Romano, J. P. and Wolf, M. (2005). "Stepwise Multiple Testing as Formalized
  Data Snooping." *Econometrica*.
- Pardo, R. (2008). *The Evaluation and Optimization of Trading Strategies*.
  Wiley. -- walk-forward methodology
