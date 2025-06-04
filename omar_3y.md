You're calculating a 3-year convexity signal, defined as:

Convexity
=
3Y Swap Rate
−
Implied Rate from Futures
Convexity=3Y Swap Rate−Implied Rate from Futures
This is essentially saying: "How rich or cheap is the 3Y swap relative to what the futures market is implying?"

The goal of the features is to:

Measure how this convexity signal behaves relative to recent trends (via EWM).

Provide indicators for mean-reversion or momentum behavior in the signal.

Decompose convexity into components associated with specific futures (white and green packs).

⚙️ Step-by-Step Feature Breakdown
1. Base Series Construction
a. Convexity
python
Copy
Edit
convexity = target_yield - self.calculate_implied_yield_from_futures(benchmark_yields)
target_yield: the 3Y swap rate.

calculate_implied_yield_from_futures(...): computes the forward-looking rate implied by futures.

Interpretation: If this value is large, the swap is rich vs futures (possibly due to convexity premium or flow pressure).

b. IMM Buckets (2y_1y_imms and 0y_1y_imms)
python
Copy
Edit
create_nyear_1year_imm_column(benchmark_yields, 2)
create_nyear_1year_imm_column(benchmark_yields, 0)
These extract columns corresponding to forward-starting 1-year IMM futures strips starting 0 or 2 years out.

Used to approximate white pack (near-term rates) and green pack (2–3Y area).

2. Swap Rate vs Smoothed Futures Packs
a. White Swap
```python
white_swap = target_yield - whites_smoothed
```

whites = average(0y_1y_imms)


Smoothed using exponential moving average (EWM) with 5-minute half-life.

Interpretation: Measures how rich/cheap the 3Y swap is relative to front-end expectations.

b. Green Swap

```
green_swap = target_yield - greens_smoothed
```

Same as above, but using the 2y-3y forward strip.

Interpretation: How much the 3Y swap deviates from mid-curve forwards (more relevant to convexity flows).

3. Feature Engineering: Distance to EWM
a. Single Distance to EWM
python
Copy
Edit
create_dist_to_ewm_feature(...)
This computes:

feature = xt - EWM(xt)

Measures how far the signal is from its smoothed value.

High positive value: signal is spiking up.

Negative value: signal is depressed.

Used for:

green_swap (50min smoothing)

white_swap (20min smoothing)

b. Double Distance to EWM
python
create_double_dist_to_ewm_feature(...)

This computes the difference between two EWM distances:

(x - emm_25(x)) - (x - emm_10(x)) = emm_10 - emm_25

Captures short-term trend relative to long-term trend.

Positive: recent values are rising faster than long-term average ⇒ momentum.

Negative: signal is pulling back ⇒ reversion.

Used on the raw convexity signal.


| Feature Name                                 | Meaning                                              | Economic Interpretation                               |
| -------------------------------------------- | ---------------------------------------------------- | ----------------------------------------------------- |
| `white_swap dist to ewm (20)min`             | Deviation of swap vs front-end pack from 20-min mean | Detects fast deviations in near-term relative pricing |
| `green_swap dist to ewm (50)min`             | Deviation of swap vs green pack from 50-min mean     | Detects medium-term mispricing in the belly           |
| `convexity double dist to ewm (10), (25)min` | 10-min EWM minus 25-min EWM of convexity             | Captures convexity momentum vs reversion              |


📌 What These Features Help With
Signal timing: When convexity is stretched relative to its recent history, it may be about to revert or continue trending.

Relative value detection: By decomposing the swap into deviations from white and green packs, you're capturing different curve segment contributions to convexity.

Feature richness: These allow your model to learn whether sharp moves or sustained trends in convexity have predictive value for returns.