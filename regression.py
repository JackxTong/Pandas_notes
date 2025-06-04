import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Assume `features` contains all your engineered features
# Choose one of the target columns (e.g., predicting future 30min diff of s1)
target_column = 's1_diff_30min'

# Drop the target from the features
X = features.drop(columns=[target_column])
y = features[target_column]

# Drop rows with NaN values (if any)
X = X.dropna()
y = y.loc[X.index]  # ensure alignment

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False  # time series, no shuffling
)

# Fit linear regression
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"Out-of-sample R²: {r2:.4f}")


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.plot(y_test.index, y_test, label='Actual', alpha=0.7)
plt.plot(y_test.index, y_pred, label='Predicted', alpha=0.7)
plt.legend()
plt.title('Actual vs Predicted: s1 30min Difference')
plt.show()
