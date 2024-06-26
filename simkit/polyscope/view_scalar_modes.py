import time
import numpy as np

import polyscope as ps


def view_scalar_modes(X, T, W, period=1/10, cmap=None):

    ps.init()
    ps.set_give_focus_on_show(True)
    dt = T.shape[1]

    if dt == 1:
        geo = ps.register_point_cloud("geo", X)
    elif dt == 2:
        geo = ps.register_curve_network("geo", X, T )
    elif dt == 3:
        geo = ps.register_surface_mesh("geo", X, T)
    elif dt == 4:
        geo = ps.register_volume_mesh("geo", X, T)

    dw = W.shape[1]



    for i in range(dw):
        Wi = W[:, i]
        wmax = np.abs(Wi).max()
        geo.add_scalar_quantity("mode " + str(i), Wi, enabled=True, vminmax=[-wmax, wmax], cmap=cmap)

        time.sleep(period)

        ps.frame_tick()

    ps.show()
    return