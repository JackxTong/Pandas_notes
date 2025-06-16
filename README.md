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


half_life_mins = [15, 20, 25, 40, 50, 60]
double_half_lifes = [(5, 10), (10, 25), (25, 40), (40, 50), (50, 60), (25, 60), (10, 60)]

new_features = pd.DataFrame()

for feature in X.columns:
    new_features[feature] = X[feature]
    
    for half_life in half_life_mins:
        feature_ewm_mean = X[feature].ewm(halflife=str(half_life) + "min", times=X[feature].index).mean()
        new_features[feature + "_dist_to_ewm_" + str(half_life) + "_min"] = X[feature] - feature_ewm_mean

    for double_half_life in double_half_lifes:
        double_lower = double_half_life[0]
        double_higher = double_half_life[1]

        lower_feature_ewm_mean = X[feature].ewm(halflife=str(double_lower) + "min", times=X[feature].index).mean()
        higher_feature_ewm_mean = X[feature].ewm(halflife=str(double_higher) + "min", times=X[feature].index).mean()

        new_features[feature + "_double_dist_to_ewm_" + str(double_half_life) + "_min"] = (
            higher_feature_ewm_mean - lower_feature_ewm_mean
        )
