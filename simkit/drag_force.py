import numpy as np



def drag_force(V_mesh, N, V_fluid, C=None):

    """
    Computes the drag force acting on the surface mesh X,T immersed in a fluid of 
    velocity V_fluid, with a drag coefficient of c 
    according to 
    
    V = (V_mesh - V_fluid)
    F = - C* ||V||^2  V / ||V|| 

    
        
    Parameters
    ----------
    V_mesh : (t, d) numpy array
        Velocities of the tets of the surface mesh (gross, should be linearly interpolated from the verts)
   
    N : (t, d) numpy array
        Normals of the tets of the surface mesh

    V_fluid : (t, d) numpy array
        Velocity of the fluid
    C : float
        Drag coefficient

    """

    return