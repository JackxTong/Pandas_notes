
import numpy as np
import pandas as pd

def one_y_three_m_forward(R_1Y: pd.Series,
                          R_1y1y: pd.Series,
                          check_R_2Y: pd.Series | None = None
                         ) -> pd.Series:
    """
    Compute the 1Y3M forward-starting par swap rate from
    1Y spot, 1Y1Y forward and (optionally) 2Y spot rates.

    All rates are expected in *decimal* form, e.g. 0.038 for 3.8 %.
    Payment frequency is assumed to be quarterly (EURIBOR-3M swaps).
    A log-linear discount-factor interpolation is used between 1Y and 2Y.

    Parameters
    ----------
    R_1Y : pd.Series
        1-year spot par swap rate.
    R_1y1y : pd.Series
        1-year-starting-in-1-year forward par swap rate.
    check_R_2Y : pd.Series, optional
        2-year spot par swap rate (used only as a diagnostic).

    Returns
    -------
    pd.Series
        1Y3M forward-starting par swap rate.
    """

    # --- Step 1 : discount factors at 1Y and 2Y (simple one-period accrual) ---
    D1 = 1.0 / (1.0 + R_1Y)                    #   D(1)
    D2 = D1 / (1.0 + R_1y1y)                   #   D(2)  =  D(1) / (1 + fwd_1y1y)

    # Diagnostic (optional): implied 2Y from D1 & D2 vs supplied R_2Y
    if check_R_2Y is not None:
        implied_R2Y = (1.0 / D2) ** 0.5 - 1.0   # simple annual-comp yield
        diff = (implied_R2Y - check_R_2Y).abs()
        if diff.max() > 1e-4:                   # 1 bp tolerance
            print("⚠️  supplied R_2Y deviates from value implied by R_1Y & R_1y1y")

    # --- Step 2 : discount factor at 1.25Y via log-linear interpolation ---
    lnD1   = np.log(D1)
    lnD2   = np.log(D2)
    lnD125 = lnD1 + 0.25 * (lnD2 - lnD1)       #   0.25 = (1.25 − 1)/(2 − 1)
    D125   = np.exp(lnD125)                    #   D(1.25)

    # --- Step 3 : par rate for a single 3-month period between 1Y and 1Y3M ---
    accrual = 0.25                             # 3 months
    R_1y3m = (D1 - D125) / (accrual * D125)

    return R_1y3m


📝
	1.	Discount factors
Treat each par swap rate as an annually-compounded yield for its whole tenor, giving
D(1) = 1 / (1 + R_{0,1}), \quad D(2) = D(1) / (1 + R_{1,1}) .
	2.	Log-linear interpolation of \ln D(t) is the standard quick-and-dirty way to fill the gap between 1 Y and 2 Y when you lack the full short-end curve.
	3.	Single-period par-swap formula
For a one–period (3-month) fixed–vs-floating swap the par rate is
R_{1\text{Y},0.25} \;=\; \frac{D(1) - D(1.25)}{0.25 \, D(1.25)} .

That’s the result returned by the function (R_1y3m), aligned on the same date index as your input Series.


