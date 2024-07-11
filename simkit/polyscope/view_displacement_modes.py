import time
import numpy as np

import polyscope as ps


def view_displacement_modes(X, T, W, a=0.1, period=100):

    ps.init()
    ps.set_give_focus_on_show(True)
    dt = T.shape[1]
    dim = X.shape[1]
    if dt == 1:
        geo = ps.register_point_cloud("geo", X)
    elif dt == 2:
        geo = ps.register_curve_network("geo", X, T )
    elif dt == 3:
        geo = ps.register_surface_mesh("geo", X, T)
    elif dt == 4:
        geo = ps.register_volume_mesh("geo", X, T)

    dw = W.shape[1]

    # for i in range(dw):
    #     Wi = W[:, i]
    #     U =  Wi.reshape((-1, dim))
    #     Umax = np.linalg.norm(W, axis=1)
            
    #     geo.add_vector_quantity("eigenvector " + str(i), U)

    #     ps.frame_tick() 


    while True:
        for i in range(dw):
            Wi = W[:, i]
            U =  Wi.reshape((-1, dim))
            Umax = np.linalg.norm(W, axis=1)
            
            for i in range(period):
                D = a* U * np.sin( 2 * np.pi * i / period)
                geo.update_vertex_positions(X + D)

                
                ps.frame_tick() 

    ps.show()
    return