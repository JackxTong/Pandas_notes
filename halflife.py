import math

print("\nHalf-Life of Mean Reversion (based on AR(1)):")
for col in diff_combined.columns:
    phi = model_results[col]["AR(1)"].params['L1']
    if 0 < phi < 1:
        half_life = math.log(0.5) / math.log(phi)
        print(f"  {col}: φ = {phi:.4f}, half-life ≈ {half_life:.2f} steps")
    else:
        print(f"  {col}: φ = {phi:.4f}, half-life not meaningful (φ ≤ 0 or ≥ 1)")
