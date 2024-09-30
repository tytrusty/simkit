import numpy as np
import scipy as sp
import igl
import polyscope as ps
import polyscope.imgui as psim
from PIL import Image
import time

from modal_impulses.normalize_and_center import normalize_and_center
from simkit.simkit.polyscope.view_clusters import view_clusters
from simkit.simkit.polyscope.view_cubature import view_cubature
from simkit.simkit.polyscope.view_displacement_modes import view_displacement_modes
from simkit.simkit.polyscope.view_scalar_modes import view_scalar_modes

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

class MeshPetWorldParams():
    def __init__(self, mesh_file, subspace_params, sim_params,name="default", pinning_method="back_z", speed=0.1, substeps = 1, k_pin=1e7, k_contact=1e7, pin_t=0.1, sphere_radius=0.5, tex_png=None, tex_obj=None):
        self.name = name
        self.mesh_file = mesh_file
        self.tex_png = tex_png
        self.tex_obj = tex_obj
        self.subspace_params = subspace_params
        self.sim_params = sim_params
        self.scale = speed
        self.substeps = substeps
        self.k_pin = k_pin
        self.k_contact = k_contact
        self.pinning_method = pinning_method
        self.pin_t = pin_t
        self.sphere_radius = sphere_radius
        pass


class MeshPet(WorldParams):
    def __init__(self, params_list: MeshPetWorldParams | list):

        if not isinstance(params_list, list):
            params_list = [params_list]
        
        self.params_list = params_list 
        
        self.params_list_names = [params.name for params in params_list]

        self.params_index = 0
        self.params = params_list[self.params_index]
        
        self.init_world_from_params( self.params)
    
     
    def init_world_from_params(self, params):


        self.timings = np.zeros((50, 1))
        self.counter = 0


        [X, T, F] = igl.read_mesh(params.mesh_file)
        X, n_translation, n_scale = normalize_and_center(X, return_params=True)
    
        self.use_tex = params.tex_png is not None and params.tex_obj is not None
        if self.use_tex:
            [V_tex, UV_tex, _, F_tex, FUV_tex, _ ] = igl.read_obj( params.tex_obj)
            V_tex += n_translation
            V_tex *= n_scale
            rgb = np.array(Image.open( params.tex_png))

            self.V_tex = V_tex
            self.UV_tex = UV_tex
            self.F_tex = F_tex
            self.FUV_tex = FUV_tex
            self.rgb = rgb

        dim = X.shape[1]
        n = X.shape[0]
        self.dim = dim
        self.X = X
        self.T = T

        self.faceI = np.unique(igl.boundary_facets(T))

        # set up external forces/quadratics
        self.bg =  - gravity_force(X, T, rho=params.sim_params.rho, a=-10).reshape(-1, 1) 

        self.sub = create_subspace(X, T, params.subspace_params)

        # set up pinned vertices... should expose this to UI/params somehow
        self.pinned, self.pinnedI = create_selection(self.params.pinning_method, self.X, self.params.pin_t)        
        

        # view_displacement_modes(self.X, igl.boundary_facets(self.T), self.sub.B)
        [self.sim, self.BQB_ext, self.Bb_ext, self.z, self.s, self.z_dot] =  self.compute_subspace_sim_quantities(X, T, self.sub, params, self.pinnedI, self.bg)
        
        ### Set up sphere collider
        self.sphere_pos = np.array([[0, 2., 0]])
        self.sphere_disp = np.array([[0, 0., 0]])
        self.sphere_radius = params.sphere_radius
                
        ## UI stuff. scale is speed at which we move spehre
        self.speed = params.scale
        self.pause = False
    
    def play(self):
        # scene visuals
        ps.init()
        ps.remove_all_structures()
        ps.set_ground_plane_mode("none")

        # visualize collider shphere
        self.sphere = ps.register_point_cloud("sphere", self.sphere_pos.reshape(1, -1))
        self.sphere.add_scalar_quantity("rad", np.array([self.sphere_radius]))
        self.sphere.set_point_radius_quantity("rad", autoscale=False)

        # visualize character mesh
        self.points = ps.register_point_cloud("points", self.X[self.sub.conI, :])

        if self.use_tex:
            self.mesh = ps.register_surface_mesh("mesh", self.V_tex, self.F_tex, edge_width=0)
            fuv = self.FUV_tex.flatten()
            uv = self.UV_tex[fuv, :]
            self.mesh.add_parameterization_quantity("test_param",  uv,
                                                defined_on='corners', enabled=True)
            self.mesh.add_color_quantity("test_vals", self.rgb[:, :, 0:3]/255,
                                        defined_on='texture', param_name="test_param",
                                        enabled=True)
        else:
            self.mesh = ps.register_surface_mesh("mesh", self.X,igl.boundary_facets(self.T), edge_width=0)
            self.mesh.add_scalar_quantity("pinned", self.pinned, enabled=True)
    
        ps.set_user_callback(self.callback)
        ps.show()
    
        return

    def contact_energy(self, z):
        Xcon = (self.sub.SB @ z).reshape(-1, self.dim)
        energy_sphere = contact_springs_sphere_energy(Xcon, self.params.k_contact, self.sphere_pos + self.sphere_disp,self.sphere_radius)
        energy_total = energy_sphere
        return energy_total

    def contact_gradient(self, z):
        Xcon = (self.sub.SB @ z).reshape(-1, self.dim)
        gradient_sphere = contact_springs_sphere_gradient(Xcon, self.params.k_contact, self.sphere_pos + self.sphere_disp, self.sphere_radius)    
        gradient_total = self.sub.SB.T @ ( gradient_sphere ) 
        return (gradient_total)

    def contact_hessian(self, z):
        Xcon = (self.sub.SB @ z).reshape(-1, self.dim)
        hessian_sphere = contact_springs_sphere_hessian(Xcon, self.params.k_contact,self.sphere_pos  + self.sphere_disp, self.sphere_radius)
        hessian_total =  self.sub.SB.T @ ( hessian_sphere) @ self.sub.SB  
        return hessian_total

    def compute_subspace_sim_quantities(self, X, T, sub, params, pinnedI, bg):
        dim = X.shape[1]
        n = X.shape[0]
        sim = ElasticROMMFEMSim(X, T, sub.B,  sub.cI, sub.cW, params.sim_params)
        
        Qpin, bpin = dirichlet_penalty(pinnedI, X[pinnedI], n, params.k_pin)

        BQB_ext = sub.B.T @ Qpin @ sub.B
        Bb_ext = sub.B.T @ (bg + bpin)
        # initialize simulation state
        z, s,z_dot = sim.rest_state()

        return sim, BQB_ext, Bb_ext, z, s, z_dot
    

    def callback(self):
        # update state, collider states
                
        start_time = time.time()
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
        
        changed, self.speed = psim.SliderFloat("Control Speed", self.speed, v_min=0.01, v_max=1)
        changed, self.sphere_radius = psim.SliderFloat("Sphere Radius", self.sphere_radius, v_min=0.1, v_max=1)
        if changed :
            self.sphere.add_scalar_quantity("rad", np.array([self.sphere_radius]))
            self.sphere.set_point_radius_quantity("rad", autoscale=False)

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


        if psim.IsKeyPressed(psim.ImGuiKey_Space):
            self.pause = not self.pause

        if psim.IsKeyPressed(psim.ImGuiKey_J):
            self.sphere_disp += self.speed * np.array([[-1, 0., 0]])
        if psim.IsKeyPressed(psim.ImGuiKey_L):
            self.sphere_disp += self.speed * np.array([[1, 0., 0]])
        if psim.IsKeyPressed(psim.ImGuiKey_I):
            self.sphere_disp += self.speed * np.array([[0, 1., 0]])
        if psim.IsKeyPressed(psim.ImGuiKey_K):
            self.sphere_disp += self.speed * np.array([[0, -1., 0]])
        if psim.IsKeyPressed(psim.ImGuiKey_U):
            self.sphere_disp += self.speed * np.array([[0, 0., 1]])
        if psim.IsKeyPressed(psim.ImGuiKey_O): 
            self.sphere_disp += self.speed * np.array([[0, 0., -1]])

        if not self.pause:
            # do a couple simulation substeps
            for _i in range(self.params.substeps):
                z_next, a_next = self.sim.step(self.z, self.s, self.z_dot, Q_ext=self.BQB_ext, b_ext=self.Bb_ext,
                                        ext_energy_func = self.contact_energy, ext_gradient_func= self.contact_gradient, ext_hessian_func = self.contact_hessian)
                
                self.z_dot = (z_next - self.z) / self.sim.params.h    
                self.z = z_next.copy()
                self.s = a_next.copy()
            
            # update geometry
            x = self.sub.B @ self.z 
            U = x.reshape(-1, self.dim)

            # update visuals 
            if self.use_tex:
                self.mesh.update_vertex_positions(U[self.faceI, :])
            else:
                self.mesh.update_vertex_positions(U)

            self.points.update_point_positions(U[self.sub.conI, :])
            self.sphere.update_point_positions((self.sphere_pos + self.sphere_disp).reshape(1, -1))


        end_time = time.time()
        total_time = end_time - start_time
        self.timings[self.counter % self.timings.shape[0]] = total_time
        self.counter += 1

        psim.Text("Time taken for sim step : " + str(self.timings.mean()))
        psim.Text("FPS : " + str(1/self.timings.mean()))
        

    def vis_subspace(self):

        view_scalar_modes(self.X, igl.boundary_facets(self.T), self.sub.W)

        