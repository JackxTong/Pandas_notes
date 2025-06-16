# Slope features within each group
slope_features = {}

# Helper to build slope features for a group
def add_slope_features(group, group_name):
    for i in range(len(group)):
        for j in range(i+1, len(group)):
            name = f'slope_{group_name}_{group[i]}_{group[j]}'
            df[name] = df[group[j]] - df[group[i]]
            slope_features[name] = df[name]

# Apply to each group
add_slope_features(group_A, 'A')
add_slope_features(group_B, 'B')
add_slope_features(group_C, 'C')
