import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# X, y: your dataset with 2000 features
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.3,     # Randomly sample 30% of features per tree
    colsample_bylevel=0.6,    # Randomly sample 60% of features per tree level
    reg_alpha=10,             # L1 regularization to push weights to 0
    reg_lambda=1,             # L2 regularization
    early_stopping_rounds=25,
    eval_metric="rmse",
    verbosity=1
)

model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          verbose=10)

y_pred = model.predict(X_val)
print("Validation R²:", r2_score(y_val, y_pred))
