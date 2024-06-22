

class kinetic_energy_precomp()
    def __init__(self, M):
        self.M = M

def kinetic_energy(x, v, X, T,  rho=1, dt=1e-2, pre=None):
    if pre is None:
        y = x + v * dt
        d = (x - y)
        M = massmatrix(self.X, self.T, self.rho)
        e = (1/dt**2 ) 0.5 * d.T @ M @ d
    else:
        e = (1/dt**2 ) 0.5 * d.T @ pre.M @ d


    return e