 More Signal Transformations
✅ First and Second Differences
Motivation: Capture rate of change and acceleration (momentum, convexity skew).

Examples:

red_signal.diff() — short-term momentum

red_signal.diff().diff() — convexity of the convexity (second derivative)

✅ Ratios
Useful when magnitudes vary and you care more about relative differences:

red_signal / green_signal

red_dist_to_ewm / red_signal — normalization of dislocation

🧮 2. Rolling Statistics (on top of smoothed signals)
Apply over 5, 10, 30-minute windows (or overlapping).

rolling_std, rolling_mean, rolling_skew, rolling_kurt on:

dist_to_ewm

signal.diff()

Example: If red is stable and green is noisy, the relative volatility spread becomes informative.

🔀 3. Cross-features: Relative Spreads and Ratios
Cross-differences:
Already doing green_dist_to_ewm - red_dist_to_ewm, but expand this:

white - green, white - red

white_signal - green_signal, not just smoothed versions

Time-decay normalized versions:
signal - long_term_ewm (e.g. half-life 30min)

short_term_ewm - long_term_ewm: Think of it like a MACD in trading.

📉 4. Event-Based & Regime Features
Time-based flags:
is_opening_hour, is_lunch_lull, is_closing_hour — used as categorical inputs

Might align well with liquidity cycles or macro announcement timing

Regime-based:
Run KMeans or GMM on rolling vol, skew, etc. to define rate regimes, then:

Add regime as categorical feature.

Build features like signal × regime.

📚 5. Domain-Aware Features
These use financial intuition:

Curve Shape:
green_signal - red_signal — slope

white_signal - green_signal — curvature

white + red - 2 * green — second derivative-style measure

Carry/Roll:
Estimate the expected roll-down over the next interval (e.g., how ER2 will change as it "becomes" ER1).

Approximated via future[i] - future[i+1], or forward-difference.

Regression Residuals:
Fit a local linear model of swaps vs futures (e.g., red swap ~ ER1-4), then use:

Residual as a feature → "unexplained portion of swap by futures"

Slope (beta) as a feature → "sensitivity"

🧠 6. Embedding or Learned Features (if you're using neural nets)
Autoencoder or PCA on:
The ER1–ER20 curve snapshot at each time → get compressed latent features.

Could capture global shifts or structural anomalies.


| Category                 | Feature Examples                                   |
| ------------------------ | -------------------------------------------------- |
| First/Second Derivatives | `signal.diff()`, `signal.diff().diff()`            |
| Rolling Stats            | `rolling_std(signal)`, `rolling_skew(dist_to_ewm)` |
| Relative Ratios          | `red/green`, `signal / ewm`                        |
| Curve Shape              | `green - red`, `white + red - 2 * green`           |
| Residuals                | `residual of swap ~ ERs`                           |
| Regime Features          | `vol_regime`, `skew_regime`, `macro_time_flag`     |
| PCA/AE                   | latent features from full ER snapshot              |



7. Nonlinear transforms:

- exp, log, sqrt, abs, tanh, etc
- ReLU-style: max(0, dist_to_ewm)
- Your curve-based features (green - red, white + red - 2*green) model curvature.
- exp(-abs(curvature)) → penalize extreme curvature more

| Feature                        | Useful Transform          | Why                              |
| ------------------------------ | ------------------------- | -------------------------------- |
| `dist_to_ewm`                  | `abs`, `square`, `sqrt`   | Model magnitude, nonlinearity    |
| `signal`                       | `tanh`, `log(1 + abs(x))` | Stabilize heavy tails            |
| `green - red`                  | `square`, `exp(-x²)`      | Emphasize big slopes             |
| `residual from swap ~ futures` | `abs`, `cube`             | Model mispricing asymmetry       |
| `volatility`                   | `log`                     | Normalize multiplicative effects |


Even more power comes from interacting nonlinear transforms:

abs(signal) * volatility

sign(signal) * (signal**2) — useful for modeling convex payoff structures

sqrt(dist_to_ewm) * slope_of_curve

These mimic how convexity impacts PnL, which is inherently a function of magnitude × direction × sensitivity.