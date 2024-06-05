import numpy as np
def ympr_to_lame(ym : float | np.ndarray , pr : float | np.ndarray):
    mu = ym / (2*(1 + pr))
    lam = ym * pr / ((1 + pr)*(1 - 2*pr))
    return mu, lam