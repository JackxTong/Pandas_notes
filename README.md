# Slope features within each group
slope_features = {}

# Helper to build slope features for a group
def add_slope_features(group, group_name):
    for i in range(len(group)):
        for j in range(




new_features_dict = {}

for feature in X.columns:
    for half_life in half_life_mins:
        ewm = X[feature].ewm(halflife=f"{half_life}min", times=X[feature].index).mean()
        new_features_dict[f"{feature}_dist_to_ewm_{half_life}min"] = X[feature] - ewm

    for lower, upper in double_half_lifes:
        ewm_lo = X[feature].ewm(halflife=f"{lower}min", times=X[feature].index).mean()
        ewm_hi = X[feature].ewm(halflife=f"{upper}min", times=X[feature].index).mean()
        new_features_dict[f"{feature}_double_dist_to_ewm_{lower}_{upper}min"] = ewm_hi - ewm_lo

# Only now create the DataFrame
new_features = pd.DataFrame(new_features_dict, index=X.index)
