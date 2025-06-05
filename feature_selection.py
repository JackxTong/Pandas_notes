import pandas as pd
from scipy.stats import pearsonr, spearmanr

# 1 .Pearson and Spearman correlation coefficients for features against a target variable

# X is your feature DataFrame, y is your target Series
def compute_feature_correlations(X: pd.DataFrame, y: pd.Series):
    pearson_corr = {}
    spearman_corr = {}
    
    for col in X.columns:
        pearson_corr[col], _ = pearsonr(X[col], y)
        spearman_corr[col], _ = spearmanr(X[col], y)
    
    # Combine into a DataFrame
    corr_df = pd.DataFrame({
        'Pearson': pd.Series(pearson_corr),
        'Spearman': pd.Series(spearman_corr)
    })
    
    return corr_df.sort_values(by='Pearson', key=lambda x: x.abs(), ascending=False)

# Example usage:
# X = pd.DataFrame(...)  # your features
# y = pd.Series(...)     # your target
# correlations = compute_feature_correlations(X, y)
# print(correlations)



# 2.  Mutual Information (MI)
# What: Measures any dependency (linear or nonlinear) between feature and target.

# Use case: Better than correlation for non-linear features.

import pandas as pd
from sklearn.feature_selection import mutual_info_regression

# X is your feature DataFrame, y is your target Series
def compute_mutual_information(X: pd.DataFrame, y: pd.Series, discrete_features='auto', random_state=0):
    # Compute mutual information
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=random_state)
    
    # Create a DataFrame with feature names and MI scores
    mi_df = pd.DataFrame({
        'Feature': X.columns,
        'Mutual Information': mi_scores
    })
    
    # Sort by MI score
    mi_df = mi_df.sort_values(by='Mutual Information', ascending=False).reset_index(drop=True)
    
    return mi_df

# Example usage:
# X = pd.DataFrame(...)  # your features
# y = pd.Series(...)     # your target
# mi_df = compute_mutual_information(X, y)
# print(mi_df)


# 3. Feature Importance using Random Forest
# Model-Based Feature Importance
# (a) Tree-Based Models
# Fit a Random Forest or Gradient Boosted Tree and use .feature_importances_

# Captures non-linear, interaction effects.

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor().fit(X, y)
importances = model.feature_importances_
importances_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False).reset_index(drop=True)


# Measure drop in performance when you randomly shuffle each feature.

# More reliable than built-in importances for some models.
from sklearn.inspection import permutation_importance
result = permutation_importance(model, X, y, n_repeats=10)
importances_perm_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': result.importances_mean
}).sort_values(by='Importance', ascending=False).reset_index(drop=True)

# LAsso
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt

# Fit LassoCV model
model = LassoCV(cv=5, random_state=0).fit(X, y)

# Get coefficients
coefs = model.coef_

# Create a DataFrame of features and their coefficients
lasso_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': coefs
})

# Filter non-zero coefficients (selected features)
selected_features = lasso_df[lasso_df['Coefficient'] != 0]

# Sort by absolute value of coefficients
selected_features = selected_features.reindex(
    selected_features['Coefficient'].abs().sort_values(ascending=False).index
).reset_index(drop=True)

print(selected_features)

# Plot top N features
N = 20  # Change this as needed
top_features = selected_features.head(N)

plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Coefficient'])
plt.gca().invert_yaxis()
plt.title('Top Lasso-Selected Features')
plt.xlabel('Coefficient')
plt.tight_layout()
plt.show()




# 4. Recursive Feature Elimination (RFE)
# recursively removes the least important features based on a model’s coefficient or feature importance.
# Start with all features.

# Train a model (e.g., linear regression, decision tree) on the current set.

# Rank features by importance (e.g., absolute value of coefficients or .feature_importances_).

# Remove the least important feature(s).

# Repeat steps 2–4 until the desired number of features is left.

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# Define estimator and RFE selector
estimator = LinearRegression()
selector = RFE(estimator, n_features_to_select=5)  # keep top 5 features

# Fit RFE
selector = selector.fit(X, y)

# Get selected feature mask
selected_mask = selector.support_

# Get selected feature names
selected_features = X.columns[selected_mask]

print("Selected features:", list(selected_features))


# Pipeline
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

pipeline = Pipeline([
    ('feature_selection', RFE(estimator=Ridge(), n_features_to_select=5)),
    ('regression', Ridge())
])

pipeline.fit(X, y)


# This allows feature selection and modeling to be combined in a single workflow, especially useful for cross-validation and grid search.

