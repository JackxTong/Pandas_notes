def dist_to_ema(series: pd.Series, half_life_mins: int):
    """Distance from EWM with specified half-life (in minutes)."""
    return series - series.ewm(halflife=f"{half_life_mins}min", times=series.index).mean()

def create_dist_to_ewm_feature(df, column_name: str, half_life_mins: int):
    """Add single distance-to-EWM feature."""
    new_col = f"{column_name}_dist_to_ewm_{half_life_mins}min"
    df[new_col] = dist_to_ema(df[column_name], half_life_mins)
    return df

def create_double_dist_to_ewm_feature(df, column_name: str, hl1: int, hl2: int):
    """Add difference between two EWM smoothings."""
    new_col = f"{column_name}_double_dist_to_ewm_{hl1}_{hl2}min"
    ema1 = dist_to_ema(df[column_name], hl1)
    ema2 = dist_to_ema(df[column_name], hl2)
    df[new_col] = ema2 - ema1  # i.e., EWM(hl1) - EWM(hl2)
    return df

# Create a copy to hold all features
features = df_diff.copy()

# List of base columns to use for features
base_cols = features.columns

# Apply single EWM features
for col in base_cols:
    features = create_dist_to_ewm_feature(features, col, half_life_mins=20)
    features = create_dist_to_ewm_feature(features, col, half_life_mins=50)

# Apply double EWM features (short vs long smoothing comparison)
for col in base_cols:
    features = create_double_dist_to_ewm_feature(features, col, hl1=10, hl2=25)
