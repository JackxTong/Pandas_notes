# Function to compute half-life from AR(1) φ coefficient
def compute_half_life(phi):
    if 0 < phi < 1:
        return math.log(0.5) / math.log(phi)
    else:
        return np.nan

# Construct features
features = pd.DataFrame(index=diff_combined.index)

for col in diff_combined.columns:
    ar1_model = model_results[col]["AR(1)"]
    phi = ar1_model.params['L1']
    half_life = int(round(compute_half_life(phi)))
    
    if not np.isnan(half_life) and half_life > 1:
        signal_base = diff_combined[col]

        # Rolling mean deviation (Z-score like feature)
        roll_mean = signal_base.rolling(window=half_life).mean()
        roll_std = signal_base.rolling(window=half_life).std()
        zscore = (signal_base - roll_mean) / roll_std
        features[f"{col}_zscore_hl"] = zscore

        # Mean reversion signal: negative z-score suggests revert long
        features[f"{col}_mean_revert_signal"] = -zscore

        # Rolling return over half-life
        features[f"{col}_return_hl"] = signal_base.diff(periods=half_life)

        # Scaled AR(1) deviation (momentum vs reversion)
        features[f"{col}_ar1_scaled"] = phi * signal_base.shift(1)
    else:
        print(f"Skipping {col} due to invalid or short half-life")
