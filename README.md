For each col in base_features:
	•	Compute a rolling (expanding) mean over the last 100 rows within the group defined by ['date_id', 'seconds_in_bucket_group', 'stock_id'].
	•	Divide that rolling mean by the current value.
	•	Cast to Float32.
	•	Name the column like "{col}_group_expanding_mean100".



for col in base_features:
    df[f'{col}_group_expanding_mean100'] = (
        df.groupby(['date_id', 'seconds_in_bucket_group', 'stock_id'])[col]
          .transform(lambda x: x.rolling(window=100, min_periods=1).mean() / x)
          .astype(np.float32)
    )






# Example slopes
df['slope_1M_3M'] = df['F_3M'] - df['F_1M']
df['slope_3M_6M'] = df['F_6M'] - df['F_3M']
df['slope_6M_12M'] = df['F_12M'] - df['F_6M']

# Curve curvature (change in slope)
df['curve_curvature_1M_6M'] = (df['F_6M'] - df['F_3M']) - (df['F_3M'] - df['F_1M'])



window_sizes = [10, 20, 40]  # You can tune these

for col in futures_cols:
    for w in window_sizes:
        df[f'{col}_dev_mean_{w}'] = df[col] - df[col].rolling(window=w, min_periods=1).mean()
        df[f'{col}_dev_median_{w}'] = df[col] - df[col].rolling(window=w, min_periods=1).median()
    # EWM
    df[f'{col}_dev_ewm'] = df[col] - df[col].ewm(span=20, adjust=False).mean()



  for col in futures_cols:
    df[f'{col}_zscore_20'] = (
        (df[col] - df[col].rolling(window=20, min_periods=1).mean()) /
        (df[col].rolling(window=20, min_periods=1).std() + 1e-6)
    )
    
    
    
for col in futures_cols:
    df[f'{col}_vol_20'] = df[col].rolling(window=20, min_periods=1).std()
    df[f'{col}_range_20'] = (
        df[col].rolling(window=20, min_periods=1).max() -
        df[col].rolling(window=20, min_periods=1).min()
    )
    
    
    
    
lags = [1, 2, 5, 10]  # lags in ticks, e.g., 30s, 60s, etc.
for col in futures_cols:
    for lag in lags:
        df[f'{col}_lag{lag}'] = df[col].shift(lag)
        df[f'{col}_return{lag}'] = df[col] - df[col].shift(lag)



# Example: short vs medium-term futures ratios
df['ratio_1M_3M'] = df['F_1M'] / (df['F_3M'] + 1e-6)
df['ratio_3M_6M'] = df['F_3M'] / (df['F_6M'] + 1e-6)
df['ratio_6M_12M'] = df['F_6M'] / (df['F_12M'] + 1e-6)
l

    


# Use a rolling or expanding window to train on past and predict next window
window_size = 60  # e.g., 60 trading days ~ 3 months

r2_scores = []
for start in range(0, len(X) - 2 * window_size, 10):
    X_train = X.iloc[start : start + window_size]
    y_train = y.iloc[start : start + window_size]
    
    X_test = X.iloc[start + window_size : start + 2 * window_size]
    y_test = y.iloc[start + window_size : start + 2 * window_size]
    
    model = LinearRegression().fit(X_train, y_train)
    r2 = model.score(X_test, y_test)
    r2_scores.append((start, r2))





 from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd

def compute_feature_scores(X: pd.DataFrame, y: pd.Series, window_size=60, step=10, lambda_weight=1.0, gamma_weight=1.0):
    """
    For each feature, compute a score:
    Score = Mean R² - λ * Std R² - γ * PSI
    """
    feature_scores = []

    for feature in X.columns:
        r2_list = []

        # Rolling R²
        for start in range(0, len(X) - 2 * window_size, step):
            X_train = X[feature].iloc[start: start + window_size].values.reshape(-1, 1)
            y_train = y.iloc[start: start + window_size]

            X_test = X[feature].iloc[start + window_size: start + 2 * window_size].values.reshape(-1, 1)
            y_test = y.iloc[start + window_size: start + 2 * window_size]

            if len(np.unique(X_train)) <= 1 or len(np.unique(X_test)) <= 1:
                continue  # skip degenerate windows

            model = LinearRegression().fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            r2_list.append(r2)

        if len(r2_list) < 2:
            continue

        mean_r2 = np.mean(r2_list)
        std_r2 = np.std(r2_list)

        # PSI between early and late periods
        psi = compute_psi(X[feature].iloc[:90].values, X[feature].iloc[-90:].values)  # Adjust length as needed

        score = mean_r2 - lambda_weight * std_r2 - gamma_weight * psi
        feature_scores.append((feature, score, mean_r2, std_r2, psi))

    return pd.DataFrame(feature_scores, columns=['Feature', 'Score', 'MeanR2', 'StdR2', 'PSI']).sort_values('Score', ascending=False)






 def rolling_zscore(X: pd.DataFrame, window: int = 30):
    """
    Rolling z-score normalization of all features.
    """
    return (X - X.rolling(window).mean()) / X.rolling(window).std()

X_norm = rolling_zscore(X, window=30)
X_norm = X_norm.dropna()
y_aligned = y.loc[X_norm.index]





    
