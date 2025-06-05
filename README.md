# Step 1: Get absolute correlation with target
corrs = new_features.corrwith(y_smooth).abs()
top_features = corrs.sort_values(ascending=False).head(50)

# Step 2: Remove correlated duplicates
uncorrelated_features = []
for f in top_features.index:
    if all(new_features[f].corr(new_features[uf]) < 0.85 for uf in uncorrelated_features):
        uncorrelated_features.append(f)

# Step 3: Try best 1, 2, 3-feature combinations
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

best_combo = None
best_r2 = -np.inf

for k in range(1, 4):
    for combo in combinations(uncorrelated_features, k):
        X = new_features[list(combo)]
        X_train, X_test, y_train, y_test = train_test_split(X, y_smooth, test_size=0.2, shuffle=False)
        model = LinearRegression(fit_intercept=False).fit(X_train, y_train)
        r2 = r2_score(y_test, model.predict(X_test))
        if r2 > best_r2:
            best_r2 = r2
            best_combo = combo

print("Best features:", best_combo)
print("R²:", best_r2)






from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Step 1: Standardize features
X_std = StandardScaler().fit_transform(new_features)

# Step 2: Run PCA
pca = PCA()
X_pca = pca.fit_transform(X_std)

# Step 3: Analyze explained variance
import matplotlib.pyplot as plt

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.title('Explained Variance by PCA Components')
plt.grid(True)
plt.show()













import xgboost as xgb
import numpy as np
import pandas as pd
import re

# Assume: X_train, y_train, new_features already defined

# Train XGBoost
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# Dump the model and parse split conditions
model_dump = xgb_model.get_booster().get_dump()

# Extract split rules: "feature_name<0.123456"
split_rules = []
for tree in model_dump:
    lines = tree.split('\n')
    for line in lines:
        match = re.search(r'(\w+)<([-\d\.e]+)', line)
        if match:
            feature = match.group(1)
            threshold = float(match.group(2))
            split_rules.append((feature, threshold))

# Aggregate: find top thresholds per feature
from collections import defaultdict
feature_thresholds = defaultdict(list)
for feature, threshold in split_rules:
    feature_thresholds[feature].append(threshold)

# Deduplicate and average
top_thresholds = {f: sorted(set(threshes))[:3] for f, threshes in feature_thresholds.items()}

# Construct ramp (ReLU) transforms
for f, thresh_list in top_thresholds.items():
    if f in new_features.columns:
        for t in thresh_list:
            new_features[f'{f}_ramp_{t:.3f}'] = np.maximum(0, new_features[f] - t)












