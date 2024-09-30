import numpy as np
import scipy as sp
import igl
import polyscope as ps
import polyscope.imgui as psim
from PIL import Image

from modal_impulses.massmatrix import massmatrix
from modal_impulses.normalize_and_center import normalize_and_center
from simkit.simkit.polyscope.view_clusters import view_clusters
from simkit.simkit.polyscope.view_cubature import view_cubature
from simkit.simkit.polyscope.view_scalar_modes import view_scalar_modes
from simkit.simkit.sims.elastic.create_elastic_sim import create_elastic_sim

from ..subspaces.create_subspace import create_subspace

from ..gravity_force import gravity_force
from ..dirichlet_penalty import dirichlet_penalty
from ..farthest_point_sampling import farthest_point_sampling
from ..selection_matrix import selection_matrix
from ..contact_springs_sphere_energy import contact_springs_sphere_energy
from ..contact_springs_sphere_gradient import contact_springs_sphere_gradient
from ..contact_springs_sphere_hessian import contact_springs_sphere_hessian
from ..common_selections import *

from ..sims.elastic import ElasticROMMFEMSim

from .World import WorldParams

pinning_methods = ["center", "back_z"]

def create_selection(name, X, t):
    if name == "center":
        pinned, pinnedI = center_indices(X, t)
    elif name == "back_z":
        pinned, pinnedI = back_z_indices(X, t)
    else:
        raise ValueError("Unknown pinning type")
    return pinned, pinnedI

class LocalAnalysisWorldParams():
    def __init__(self, mesh_file, subspace_params, sim_params,name="default", pinning_method="back_z", speed=0.1, substeps = 1, k_pin=1e7, pin_t=0.1, interactive=False, ag=0):
        self.name = name
        self.mesh_file = mesh_file
        self.subspace_params = subspace_params
        self.sim_params = sim_params
        self.substeps = substeps
        self.k_pin = k_pin
        self.pinning_method = pinning_method
        self.pin_t = pin_t
        self.interactive = interactive
        self.ag = 0
        pass

class LocalAnalysis(WorldParams):
    def __init__(self, params: LocalAnalysisWorldParams ):
        self.params = params        
        self.init_world_from_params( self.params)
    
    def init_world_from_params(self, params):
        [X, T, F] = igl.read_mesh(params.mesh_file)
        X, n_translation, n_scale = normalize_and_center(X, return_params=True)    

        dim = X.shape[1]
        n = X.shape[0]
        self.dim = dim
        self.X = X
        self.T = T

        self.faceI = np.unique(igl.boundary_facets(T))

        # set up external forces/quadratics
        self.bg =  - gravity_force(X, T, rho=params.sim_params.rho, a=params.ag).reshape(-1, 1) 

        self.sub = create_subspace(X, T, params.subspace_params)

        # set up pinned vertices... should expose this to UI/params somehow
        self.pinned, self.pinnedI = create_selection(self.params.pinning_method, self.X, self.params.pin_t)        
            
        [self.sim, self.BQB_ext, self.Bb_ext, self.z, self.s, self.z_dot] =  self.compute_subspace_sim_quantities(X, T, self.sub, params, self.pinnedI, self.bg)
    
    def play(self):
        # scene visuals
        ps.init()
        ps.remove_all_structures()
        ps.set_ground_plane_mode("none")

        # visualize character mesh
        self.mesh = ps.register_surface_mesh("mesh", self.X,igl.boundary_facets(self.T), edge_width=0)
        self.mesh.add_scalar_quantity("pinned", self.pinned, enabled=True)

        ps.set_user_callback(self.callback)
        ps.show()
    
        return

    def compute_subspace_sim_quantities(self, X, T, sub, params, pinnedI, bg):
        dim = X.shape[1]
        n = X.shape[0]

        sim = create_elastic_sim(X, T, params.sim_params, sub=sub)
        # sim = ElasticROMMFEMSim(X, T, sub.B,  sub.cI, sub.cW, params.sim_params)
        
        Qpin, bpin = dirichlet_penalty(pinnedI, X[pinnedI], n, params.k_pin)

        BQB_ext = sub.B.T @ Qpin @ sub.B
        Bb_ext = sub.B.T @ (bg + bpin)
        # initialize simulation state
        z, s,z_dot = sim.rest_state()

        return sim, BQB_ext, Bb_ext, z, s, z_dot
    

    def callback(self):
        # update state, collider states
        
        psim.PushItemWidth(100)
        changed = psim.BeginCombo("World Params", self.params_list_names[self.params_index])
        if changed:
            for i, val in enumerate(self.params_list_names):
                _, selected = psim.Selectable(val,  self.params_list_names==val)
                if selected:
                    self.params_index = i
                    self.params = self.params_list[i]
                    self.init_world_from_params( self.params)
                    # should recompute
            psim.EndCombo()
        psim.PopItemWidth()
        
        if(psim.Button("Reset")):
            self.z, self.s, self.z_dot = self.sim.rest_state()
      
        changed, self.params.pin_t = psim.SliderFloat("Pin t", self.params.pin_t, v_min=0, v_max=1)
        if changed:
            pinned, pinnedI = create_selection(self.params.pinning_method, self.X, self.params.pin_t)        
            Qpin, bpin = dirichlet_penalty(pinnedI, self.X[pinnedI], self.X.shape[0], self.params.k_pin)
            self.BQB_ext = self.sub.B.T @ Qpin @ self.sub.B
            self.Bb_ext = self.sub.B.T @ (self.bg + bpin)
            self.mesh.add_scalar_quantity("pinned", pinned, enabled=True)

        psim.PushItemWidth(100)
        changed = psim.BeginCombo("Pinning Method", self.params.pinning_method)
        if changed:
            for val in  pinning_methods:
                _, selected = psim.Selectable(val, self.params.pinning_method==val)
                if selected:
                    self.params.pinning_method = val
                    pinned, pinnedI = create_selection(self.params.pinning_method, self.X, self.params.pin_t)        
                    Qpin, bpin = dirichlet_penalty(pinnedI, self.X[pinnedI], self.X.shape[0], self.params.k_pin)
                    self.BQB_ext = self.sub.B.T @ Qpin @ self.sub.B
                    self.Bb_ext = self.sub.B.T @ (self.bg + bpin)
                    self.mesh.add_scalar_quantity("pinned", pinned, enabled=True)
                    # should recompute
            psim.EndCombo()
        psim.PopItemWidth()


        self.z, self.s, self.z_dot = self.step_sim(self.z, self.s,  self.z_dot)
        # update geometry
        x = self.sub.B @ self.z 
        U = x.reshape(-1, self.dim)

        self.mesh.update_vertex_positions(U)

    
    def step_sim(self, z, s, z_dot):
        # do a couple simulation substeps
        for _i in range(self.params.substeps):
            z_next, s_next = self.sim.step(z, s, z_dot, Q_ext=self.BQB_ext, b_ext=self.Bb_ext)
            
            z_dot = (z_next - z) /self.sim.params.h    
            z = z_next.copy()
            s = s_next.copy()
        return z,  s, z_dot
            

    def analyze_impulse_response(self, sI, tI, Fmin=-1e13, Fmax=1e13, buckets=11):
        # analyze the simulation with a histogram of forces applied on sI and a histogram of responses on tI

        dim = self.X.shape[1]
        n = self.X.shape[0]


        mag = np.linspace(Fmin, Fmax, buckets)
        mag = np.tile(mag, (1, dim)).T.flatten()
        # apply a bunch of external forces in X, Y and Z directions for sI

        Me = sp.sparse.kron(massmatrix(self.X, self.T), sp.sparse.eye(dim))
        BMe = self.sub.B.T @ Me
        Z = np.zeros((self.sub.B.shape[1], buckets*self.dim))
        
        # import polyscope as ps
        
        for i in range(dim):
            for j in range(buckets):
                F = np.zeros((n, dim))
                F[sI, i] = mag[j]
                f = BMe @ F.reshape(-1, 1)
                
                z,s, z_dot = self.sim.rest_state()
                z0 = z.copy()
                # ps.init()
                # mesh = ps.register_surface_mesh("mesh", self.X, igl.boundary_facets(self.T), edge_width=0)
                # point = ps.register_point_cloud("point", self.X[sI].reshape(1, -1))
                # point.add_vector_quantity("force", F[[sI]])
                # ps.show()
                # do a couple simulation substeps
                for _i in range(1):
                    z_next, s_next = self.sim.step(z, s, z_dot, Q_ext=self.BQB_ext, b_ext=self.Bb_ext - f)

                    z_dot = (z_next - z) /self.sim.params.h    
                    z = z_next.copy()
                    s = s_next.copy()

                    # only apply this force on first timestep
                    f *= 0

                    # mesh.update_vertex_positions((self.sub.B @ z).reshape(-1, dim))    
                    # # ps.show()
                    # ps.frame_tick()
                Z[:, [ buckets*i  + j]] =  z

        U = self.sub.B @ Z
        U = U.reshape(-1, dim, dim , buckets)

        return U


        # D = self.sub.B @ Z
        # U2= np.zeros(U.shape)
        # for i in range(dim):
        #     for j in range(dim):
        #         I = np.arange(n) * dim + i
        #         J = np.arange(buckets) + buckets * j
        #         U2[:, i, j, :] = D[I, :][:, J ]



            
