# Slope features within each group
slope_features = {}

# Helper to build slope features for a group
def add_slope_features(group, group_name):
    for i in range(len(group)):
        for j in range(i+1, len(group)):
            name = f'slope_{group_name}_{group[i]}_{group[j]}'
            df[name] = df[group[j]] - df[group[i]]
            slope_features[name] = df[name]



slope_features = {}

# Across Group A → B
for a in group_A:
    for b in group_B:
        name = f'slope_{a}_{b}'
        df[name] = df[b] - df[a]
        slope_features[name] = df[name]

# Across Group A → C
for a in group_A:
    for c in group_C:
        name = f'slope_{a}_{c}'
        df[name] = df[c] - df[a]
        slope_features[name] = df[name]

# Across Group B → C
for b in group_B:
    for c in group_C:
        name = f'slope_{b}_{c}'
        df[name] = df[c] - df[b]
        slope_features[name] = df[name]
