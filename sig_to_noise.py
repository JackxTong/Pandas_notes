print("\nSignal-to-Noise Ratio (SNR) based on AR(1):")
for col in diff_combined.columns:
    model = model_results[col]["AR(1)"]
    signal_var = np.var(model.fittedvalues)
    noise_var = np.var(model.resid)
    snr = signal_var / noise_var if noise_var > 0 else np.nan
    print(f"  {col}: SNR ≈ {snr:.2f}")
