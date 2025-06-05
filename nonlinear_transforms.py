import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Define your selected base features
base_features = [
    'whitesdist_to_ewm_15_min',
    'redsdist_to_ewm_25_min',
    'greensdist_to_ewm_25_min'
]

# Define transform functions
transform_functions = {
    'identity': lambda x: x,
    'square': lambda x: np.power(x, 2),
    'sqrt': lambda x: np.sqrt(np.abs(x)),
    'log1p': lambda x: np.log1p(np.abs(x)),
    'exp': lambda x: np.exp(x.clip(upper=5)),  # avoid overflow
    'tanh': lambda x: np.tanh(x),
    'abs': lambda x: np.abs(x)
}

# Prepare a DataFrame to store results
results = []

# Iterate through all combinations of transformations per feature
for transform_name_1, f1 in transform_functions.items():
    for transform_name_2, f2 in transform_functions.items():
        for transform_name_3, f3 in transform_functions.items():
            # Apply transformations
            X_transformed = pd.DataFrame({
                f"{base_features[0]}__{transform_name_1}": f1(new_features[base_features[0]]),
                f"{base_features[1]}__{transform_name_2}": f2(new_features[base_features[1]]),
                f"{base_features[2]}__{transform_name_3}": f3(new_features[base_features[2]])
            })

            # Time-series split
            X_train, X_test, y_train, y_test = train_test_split(
                X_transformed, y_30min_smoothed, test_size=0.2, shuffle=False
            )

            # Fit linear model
            model = LinearRegression(fit_intercept=False)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Compute R²
            r2 = r2_score(y_test, y_pred)

            results.append({
                'transforms': (transform_name_1, transform_name_2, transform_name_3),
                'r2_score': r2
            })

# Show best N combos
results_df = pd.DataFrame(results).sort_values('r2_score', ascending=False)
import ace_tools as tools; tools.display_dataframe_to_user(name="Nonlinear Transform Search Results", dataframe=results_df)
