import numpy as np


def backtracking_line_search(f,x0,g,dx,alpha=0.01,beta=0.5,max_iter=100, threshold=1e-12):
    """
    Backtracking line search from  "Convex Optimization"[Boyd 2006] Chapter 11 "Interior-point methods" Algorithm 9.2

    Parameters
    ----------
    f : function
        Objective function.
    x0 : array
        Initial point.
    g : array
        Gradient at x0.
    dx : array
        Direction.
    alpha : float, optional
        Step size. The default is 0.01.
    beta : float, optional
        Reduction factor. The default is 0.5.
    max_iter : int, optional
        Maximum number of iterations. The default is 100.
    threshold : float, optional
        Threshold. The default is 1e-12.

    """
    assert(alpha>0 and alpha<=0.5);
    assert(beta>0 and beta<1);

    assert(np.ndim(x0) == np.ndim(dx));
    t = 1;
    fx0 = f(x0);
    for iter in range(max_iter):
        x = x0+t*dx;
        fx = f(x);
        if fx<= fx0 + alpha*t*(g.T @ dx) + threshold:
            return t, x, fx;

        t = beta*t;

    t = 0;
    x = x0;
    fx = fx0;

    return t, x, fx
