import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from xgb import XGBRegressor

# X, y: your dataset with 2000 features
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def init_xgb():
    xgb_model = XGBRegressor(
        n_estimators=1000,            # More rounds allowed
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.3,
        colsample_bylevel=0.6,
        reg_alpha=10,                # L1 regularization
        reg_lambda=1,                # L2 regularization
        early_stopping_rounds=100,   # More patience before stopping
        eval_metric="rmse",
        verbosity=1,                 # Show training progress
        random_state=42              # For reproducibility
    )
    return xgb_model

model = init_xgb()

model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          verbose=10)

y_pred = model.predict(X_val)
print("Validation R²:", r2_score(y_val, y_pred))
