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





    
