# Slope features within each group
slope_features = {}

# Helper to build slope features for a group
def add_slope_features(group, group_name):
    for i in range(len(group)):
        for j in range(i+1, len(group)):
            name = f'slope_{group_name}_{group[i]}_{group[j]}'
            df[name] = df[group[j]] - df[group[i]]
            slope_features[name] = df[name]



def add_smoothed_features(df, cols, window="5min", halflife_min=5):
    """
    Adds rolling median and EWM smoothed versions of given columns to the dataframe.
    
    Parameters:
        df (pd.DataFrame): The original dataframe with time-indexed data
        cols (list of str): List of column names to smooth
        window (str): Rolling window size (default '5min')
        halflife_min (int): Half-life for EWM in minutes (default 5)
    
    Returns:
        df (pd.DataFrame): Modified dataframe with new smoothed columns
    """
    for col in cols:
        df[f"{col}Smoothed"] = df[col].rolling(window=window).median()
        df[f"{col}Smoothed2"] = df[col].ewm(
            halflife=f"{halflife_min}min", times=df.index
        ).mean()
    return df
