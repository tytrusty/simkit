import numpy as np


def process_tet_field(mu, T):
    if mu is None:
        mu = np.ones(T.shape[0])
    elif isinstance(mu, (int, float)):
        mu = mu * np.ones(T.shape[0])
    elif isinstance(mu, str):
        mu = np.load(mu).reshape(-1)

    mu = mu.reshape(-1)
    assert mu.shape[0] == T.shape[0]
    return mu