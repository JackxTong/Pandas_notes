
import pandas as pd
import numpy as np
import pickle

# 1️⃣ Load data
df_future_yield = pd.read_csv('data\\future_yield_pivot.csv', index_col=0)
df_swap_yield = pd.read_csv('data\\swap_yield_pivot.csv', index_col=0)

# 2️⃣ Load hedge ratios
with open('data\\hedge_ratio.pkl', 'rb') as fp:
    hedge_ratios = pickle.load(fp)

# 3️⃣ Get common timestamps
timestamps = df_swap_yield.index.intersection(df_future_yield.index)
T = len(timestamps)

# 4️⃣ Determine swaps and futures
swap_fidos = sorted(list(set().union(*[set(d.keys()) for d in hedge_ratios.values()])))
future_ers = df_future_yield.columns.tolist()
S = len(swap_fidos)
N = len(future_ers)

# 5️⃣ Build the 3D weights array: (T, S, N)
weights_array = np.zeros((T, S, N))

for t_idx, d in enumerate(timestamps):
    date = d[:10]
    if date in hedge_ratios:
        hedges = hedge_ratios[date]
        for s_idx, fido in enumerate(swap_fidos):
            future_weights = hedges.get(fido, {})
            for n_idx, er in enumerate(future_ers):
                weights_array[t_idx, s_idx, n_idx] = future_weights.get(er, 0)

# 6️⃣ Convert future yields to array (T, N)
future_yields_array = df_future_yield.loc[timestamps, future_ers].values  # shape (T, N)

# 7️⃣ Compute implied yield: (T, S)
# Using Einstein summation (np.einsum) for full vectorization
implied_yields_array = np.einsum('tsn,tn->ts', weights_array, future_yields_array)

# 8️⃣ Convert swap yields to array (T, S)
# Align swap_yield columns with swap_fidos
df_swap_yield_sub = df_swap_yield.loc[timestamps, df_swap_yield.columns.intersection([str(fid) for fid in swap_fidos])]
df_swap_yield_sub = df_swap_yield_sub.reindex(columns=[str(fid) for fid in swap_fidos], fill_value=np.nan)
swap_yields_array = df_swap_yield_sub.values  # shape (T, S)

# 9️⃣ Compute convexities: (T, S)
convexities_array = swap_yields_array - implied_yields_array

# 🔟 Convert to DataFrame
df_convexities = pd.DataFrame(convexities_array, index=timestamps, columns=swap_fidos)
print(df_convexities.head())

# 🎉 Done! You now have the (T, S) convexity matrix, fully vectorized!
