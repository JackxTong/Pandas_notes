import pandas as pd



def create_past_deviation(df, feature_list):
    df_copy = df.copy()
    new_columns = {}

    for feature in feature_list:
        # Group by date and compute expanding mean or median (use cumsum/count for mean)
        def calc_deviation(group):
            expanding = group[feature].expanding().median().shift(1)  #
            return group[feature] - expanding

        deviation_col_name = f"past_deviation_from_{'median'}_{feature}"
        new_columns[deviation_col_name] = df_copy.groupby("date", group_keys=False).apply(calc_deviation)

    result_df = pd.concat([df_copy, pd.DataFrame(new_columns)], axis=1)
    return result_df

# note group[feature].expanding() builds rolling median

# s = pd.Series([3, 5, 7, 9])
# s.expanding().mean()
# Output:
# 0    3.0
# 1    4.0
# 2    5.0
# 3    6.0

# expanding().mean().shift(1) does not include current row
# Row 0 gets NaN (no past data)

# Row 1 gets the value from row 0

# Row 2 gets mean of rows 0–1

# Row 3 gets mean of rows 0–2

# … and so on
