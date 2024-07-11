import polyscope as ps
import numpy as np



def view_clusters(X, T, l):


    ps.init()
    mesh = ps.register_surface_mesh("mesh", X, T)


    mesh.add_scalar_quantity("cluster", l.astype(np.float64),defined_on='faces', cmap='rainbow')
    ps.show()

    return
