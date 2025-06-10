import pandas as pd

# Example DataFrame (replace with your actual data)
# Let's create some dummy data for demonstration
dates = pd.to_datetime(pd.date_range(start='2023-01-01 09:00:00', end='2023-01-02 16:30:00', freq='30S'))
df = pd.DataFrame({'convexity': range(len(dates))}, index=dates)

# Calculate the raw difference
df['convexity_diff_raw'] = df['convexity'].shift(-60) - df['convexity'] # -60 for 30 mins forward

# Define the market close time (e.g., 16:30:00)
market_close_time = pd.to_datetime('16:30:00').time()

# Calculate the time 30 minutes before market close
# This is the last valid time for which we can calculate a 30-min forward diff
last_valid_time_for_diff = (pd.to_datetime('16:30:00') - pd.Timedelta(minutes=30)).time() # 16:00:00

# Create a mask: True for rows where the time component is after the last valid time
mask = df.index.time >= last_valid_time_for_diff

# Apply the mask to set the difference to NaN for those rows
df.loc[mask, 'convexity_diff_raw'] = pd.NA # Or np.nan, depending on your pandas version
