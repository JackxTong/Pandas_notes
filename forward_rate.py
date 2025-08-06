import numpy as np

def forward_rate_from_spot(R1: float, R2: float, T1: float) -> float:
    """
    Compute forward rate F(t, T1, T2=T1+0.25) using linear interpolation
    of zero rates between 1Y and 2Y.

    Parameters
    ----------
    R1 : float
        1Y spot zero rate (in decimals).
    R2 : float
        2Y spot zero rate (in decimals).
    T1 : float
        Start of forward period (must be in (1, 2)).

    Returns
    -------
    float
        Forward rate over [T1, T1+0.25] in simple (Act/Act) terms.
    """
    T2 = T1 + 0.25

    # Interpolated zero rates at T1 and T2
    r_T1 = R1 + (T1 - 1.0) * (R2 - R1)
    r_T2 = R1 + (T2 - 1.0) * (R2 - R1)

    # Discount factors
    D_T1 = np.exp(-r_T1 * T1)
    D_T2 = np.exp(-r_T2 * T2)

    # Simple forward rate between T1 and T2
    F = (D_T1 - D_T2) / (0.25 * D_T2)

    return F