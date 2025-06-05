from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Time-series split
X_train, X_test, y_train, y_test = train_test_split(
    manual_features, y_30min_smoothed, test_size=0.2, shuffle=False
)

# XGBoost model
xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# early stopping
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=10,
    verbose=False
)

xgb_model.fit(X_train, y_train)

# Prediction and scoring
y_pred = xgb_model.predict(X_test)
print(f"XGB R²: {r2_score(y_test, y_pred):.4f}")

from lightgbm import LGBMRegressor

lgb_model = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
lgb_model.fit(X_train, y_train)

y_pred = lgb_model.predict(X_test)
print(f"LGBM R²: {r2_score(y_test, y_pred):.4f}")
