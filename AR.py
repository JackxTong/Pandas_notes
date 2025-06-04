from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

model_results = {}

for col in diff_combined.columns:
    print(f"\nModeling {col}:")

    # Fit AR(1)
    ar1_model = AutoReg(diff_combined[col], lags=1).fit()
    print(f"  AR(1) coef: {ar1_model.params['L1']:.4f}")
    
    # Fit AR(5)
    ar5_model = AutoReg(diff_combined[col], lags=5).fit()
    print(f"  AR(5) coefs: {ar5_model.params.values[1:]}")
    
    # Fit ARMA(1,1) using ARIMA(1,0,1)
    arma11_model = ARIMA(diff_combined[col], order=(1, 0, 1)).fit()
    print(f"  ARMA(1,1) AR coef: {arma11_model.params['ar.L1']:.4f}, MA coef: {arma11_model.params['ma.L1']:.4f}")
    
    # Store models for comparison
    model_results[col] = {
        "AR(1)": ar1_model,
        "AR(5)": ar5_model,
        "ARMA(1,1)": arma11_model,
    }
