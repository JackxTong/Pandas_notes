import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, Ridge, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ----- Step 1: VIF -----
# If X is not a DataFrame, convert
X_df = pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])

vif_data = pd.DataFrame()
vif_data["feature"] = X_df.columns
vif_data["VIF"] = [variance_inflation_factor(X_df.values, i) for i in range(X_df.shape[1])]
print("=== VIF ===")
print(vif_data.sort_values("VIF", ascending=False).reset_index(drop=True))

# ----- Step 2: TimeSeriesSplit -----
tscv = TimeSeriesSplit(n_splits=5)

# ----- Step 3: Model definitions -----
models = {
    "Lasso": LassoCV(alphas=np.logspace(-3, 1, 50), cv=tscv, max_iter=10000),
    "ElasticNet": ElasticNetCV(alphas=np.logspace(-3, 1, 50), l1_ratio=[0.1, 0.5, 0.9], cv=tscv, max_iter=10000),
    "Ridge": GridSearchCV(Ridge(), param_grid={"alpha": np.logspace(-3, 1, 50)}, cv=tscv)
}

# ----- Step 4: Fit and interpret -----
for name, model in models.items():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", model)
    ])
    pipe.fit(X, y)

    # Extract coefficients
    if name == "Ridge":
        reg_model = pipe.named_steps["reg"].best_estimator_
    else:
        reg_model = pipe.named_steps["reg"]

    coefs = reg_model.coef_
    r2 = pipe.score(X, y)
    nonzero_count = np.sum(coefs != 0)

    # Link coefficients to features
    coef_df = pd.DataFrame({
        "feature": X_df.columns,
        "coefficient": coefs
    }).sort_values(by="coefficient", key=np.abs, ascending=False)

    print(f"\n=== {name} ===")
    print(f"R² Score: {r2:.4f}")
    print(f"Non-zero Coefficients: {nonzero_count}")
    print("Top features by absolute weight:")
    print(coef_df.head(10).to_string(index=False))
