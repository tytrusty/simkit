


def normalize_and_center(X, return_params=False):

    # U = X.copy()

    t = -X.mean(axis=0)
    X += t
    scale = 2 / max(X.max(axis=0) - X.min(axis=0))
    X *= scale


    if return_params :
        return X, t, scale
    else:
        return X

