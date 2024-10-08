import numpy as np


def process_delta(delta, T):
    if delta is None:
        delta = 0 * np.zeros((T.shape[0], 1), dtype=int)
    elif isinstance(delta, (int)):
        delta = delta * np.ones((T.shape[0], 1), dtype=int)
    elif isinstance(delta, str):
        delta = np.load(delta)
    assert (delta.shape[0] == T.shape[0])

    return delta