import numpy as np
from sklearn.linear_model import LinearRegression

def compute_vif_manual(X):
    """Compute VIF for each feature in X manually without using statsmodels."""
    X = np.asarray(X)
    n_features = X.shape[1]
    vif = []

    for i in range(n_features):
        X_i = X[:, i]
        X_rest = np.delete(X, i, axis=1)
        
        # Fit regression model for X_i ~ X_rest
        model = LinearRegression().fit(X_rest, X_i)
        r2 = model.score(X_rest, X_i)
        
        vif_i = 1 / (1 - r2) if r2 < 1 else np.inf
        vif.append(vif_i)

    return vif



vif_scores = compute_vif_manual(X)
for i, score in enumerate(vif_scores):
    print(f"Feature {i}: VIF = {score:.2f}")



import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Assume X and y are your input features and target
# X should be a 2D NumPy array or DataFrame
# y should be a 1D NumPy array or Series

# Convert to DataFrame for VIF computation
X_df = pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])

# Compute VIF for each feature
vif_data = pd.DataFrame()
vif_data["feature"] = X_df.columns
vif_data["VIF"] = [variance_inflation_factor(X_df.values, i) for i in range(X_df.shape[1])]

print("=== Variance Inflation Factors (VIF) ===")
print(vif_data.sort_values("VIF", ascending=False).reset_index(drop=True))

# Prepare regularized regression models
models = {
    "Lasso": LassoCV(cv=5, random_state=0),
    "Ridge": RidgeCV(cv=5),
    "ElasticNet": ElasticNetCV(cv=5, random_state=0)
}

pipe = make_pipeline(StandardScaler(), ("reg", model))
...
coefs = pipe.named_steps["reg"].coef_

# Fit and evaluate each model
for name, model in models.items():
    pipe = make_pipeline(StandardScaler(), model)
    pipe.fit(X, y)
    r2 = pipe.score(X, y)
    coefs = pipe.named_steps[name.lower()].coef_
    nonzero_count = np.sum(coefs != 0)
    
    print(f"\n=== {name} Regression ===")
    print(f"R² Score: {r2:.4f}")
    print(f"Non-zero Coefficients: {nonzero_count}")
