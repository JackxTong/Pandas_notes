change

```python
df['whites'] = df[white_futures].mean(axis=1)  # Instead of .sum()/4
df['reds'] = df[red_futures].mean(axis=1)
df['greens'] = df[green_futures].mean(axis=1)
```

Code to create all butterfly (fly) combinations of adjacent 3-future groups:

```python
fly_features = {}

# Futures from ER1 to ER8
future_cols = [f'ER{i}' for i in range(1, 9)]

# Generate butterfly features: ERi - 2*ER(i+1) + ER(i+2)
for i in range(len(future_cols) - 2):
    name = f'{future_cols[i]}{future_cols[i+1]}{future_cols[i+2]}'
    df[name] = (
        df[future_cols[i]] 
        - 2 * df[future_cols[i+1]] 
        + df[future_cols[i+2]]
    )


```

Double flies (2nd order curvature)
Example: er1 - 2er3 + er5
```
for i in range(len(future_cols) - 4):
    name = f'{future_cols[i]}_{future_cols[i+2]}_{future_cols[i+4]}_dblfly'
    df[name] = df[future_cols[i]] - 2 * df[future_cols[i+2]] + df[future_cols[i+4]]

```


```
df['front_avg'] = df[['ER1', 'ER2', 'ER3']].mean(axis=1)
df['belly_avg'] = df[['ER4', 'ER5', 'ER6']].mean(axis=1)
df['back_avg'] = df[['ER7', 'ER8']].mean(axis=1)

df['front_belly_slope'] = df['front_avg'] - df['belly_avg']
df['belly_back_slope'] = df['belly_avg'] - df['back_avg']
```