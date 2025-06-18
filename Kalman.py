import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Kalman Smoother function ---
def kalman_smoother_1d(obs, process_var=1e-2, measurement_var=1.0):
    n = len(obs)
    x_hat = np.zeros(n)
    P = np.zeros(n)
    x_hat[0] = obs[0]
    P[0] = 1.0

    for t in range(1, n):
        # Predict
        x_hat_minus = x_hat[t-1]
        P_minus = P[t-1] + process_var

        # Update
        K = P_minus / (P_minus + measurement_var)
        x_hat[t] = x_hat_minus + K * (obs[t] - x_hat_minus)
        P[t] = (1 - K) * P_minus

    return pd.Series(x_hat, index=obs.index)

# --- 2. Apply Kalman smoother ---
X["ConvexityKalman"] = kalman_smoother_1d(X["convexity"])

# --- 3. Create lag features ---
X["KalmanLag1"] = X["ConvexityKalman"].diff(1)
X["KalmanLag5"] = X["ConvexityKalman"].diff(5)
X["KalmanLag10"] = X["ConvexityKalman"].diff(10)

# --- 4. Create 30-minute return from Kalman smoothed signal ---
# Assuming 5-minute data: 6 steps = 30 minutes
Y_kalman = X["ConvexityKalman"].diff(6)

# --- 5. Plot comparison with original target Y ---
plt.figure(figsize=(12, 4))
plt.plot(Y, label="Original Target Y (EWM 30min diff)", alpha=0.6)
plt.plot(Y_kalman, label="Kalman-smoothed 30min diff", alpha=0.8)
plt.title("Comparison of 30min Return Target: EWM vs Kalman Smoothed")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
