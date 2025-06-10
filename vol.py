import pandas as pd
import numpy as np

def add_realized_volatility_features(df, col, timestamp_col='timestamp'):
    """
    Adds past realized volatility (rolling std dev) features to a DataFrame.
    
    Parameters:
    - df: input DataFrame
    - col: name of the column on which to compute volatility
    - timestamp_col: name of the column containing datetime
    
    Returns:
    - df: same DataFrame with 3 new columns added
    """

    df = df.copy()

    # Ensure timestamp is datetime and sorted
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col)
    df.set_index(timestamp_col, inplace=True)

    # Define rolling windows in minutes
    windows = [5, 15, 30]
    for w in windows:
        vol_col = f"realized_vol_{w}min"
        df[vol_col] = (
            df[col]
            .rolling(f'{w}min')
            .std()
        )

    df.reset_index(inplace=True)
    return df
