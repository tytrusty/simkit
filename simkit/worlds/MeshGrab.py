import os
import time
import numpy as np
import scipy as sp
import igl
import polyscope as ps
import polyscope.imgui as psim
from PIL import Image

from .World import WorldParams

from ..subspaces.create_subspace import create_subspace
from ..sims.elastic.ElasticROMPDSim import ElasticROMPDSim

from ..normalize_and_center import normalize_and_center
from ..gravity_force import gravity_force
from ..dirichlet_penalty import dirichlet_penalty
from ..common_selections import create_selection


class MeshGrabWorldParams():
    def __init__(self, mesh_file, subspace_params, sim_params,name="default", pinning_method="back_z", speed=0.1, substeps = 1, k_picked=1e8, k_pin=1e7, k_contact=1e7, pin_t=0.1, sphere_radius=0.5, tex_png=None, tex_obj=None):
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
        self.k_picked = k_picked
        pass


class MeshGrab(WorldParams):
    def __init__(self, params_list: MeshGrabWorldParams | list):

        if not isinstance(params_list, list):
            params_list = [params_list]
        
        self.params_list = params_list 
        
        self.params_list_names = [params.name for params in params_list]

        self.params_index = 0
        self.params = params_list[self.params_index]
        
        self.init_world_from_params( self.params)

    

    def init_world_from_params(self, params):
        
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
        [self.sim, self.BQB_ext, self.Bb_ext, self.z,  self.z_dot] =  self.compute_subspace_sim_quantities(X, T, self.sub, params, self.pinnedI, self.bg)
        # self.z += np.random.rand(self.z.shape[0], 1)
        ### Set up sphere collider
        self.bI = None
        self.bc = None

        if sp.sparse.issparse(self.sim.B):
            self.Q_picked = sp.sparse.csc_matrix((self.sim.B.shape[1], self.sim.B.shape[1]))
        else:
            self.Q_picked = np.zeros((self.sim.B.shape[1], self.sim.B.shape[1]))
        self.b_picked = np.zeros((self.sim.B.shape[1], 1))
                
        ## UI stuff. scale is speed at which we move spehre
        self.speed = params.scale
        self.pause = False

    
    def play(self):

        self.timings = np.zeros((50, 1))
        self.counter = 0
        # scene visuals
        ps.init()
        ps.remove_all_structures()
        ps.set_ground_plane_mode("none")

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


    def compute_subspace_sim_quantities(self, X, T, sub, params, pinnedI, bg):
        n = X.shape[0]
        Qpin, bpin = dirichlet_penalty(pinnedI, X[pinnedI], n, params.k_pin)
        BQB_ext = sub.B.T @ Qpin @ sub.B
        Bb_ext = sub.B.T @ (bg + bpin)
        params.sim_params.Q0 = BQB_ext
        sim = ElasticROMPDSim(X, T, sub.B,  sub.labels, params.sim_params)

        z, z_dot = sim.rest_state()
        return sim, BQB_ext, Bb_ext, z,  z_dot
    

    def callback(self):
        start_time = time.time()
        # update state, collider states
        psim.Text("Num Vertices : " + str(self.X.shape[0]))
        psim.Text("Num Elements : " + str(self.T.shape[0]))
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
            self.z, self.z_dot = self.sim.rest_state()
        

        changed, self.speed = psim.SliderFloat("Control Speed", self.speed, v_min=0.01, v_max=1)

        if psim.IsKeyPressed(psim.ImGuiKey_Space):
            self.pause = not self.pause

        if psim.IsMouseClicked(0):
            name, i= ps.get_selection()
            print(name)
            P = (self.sim.B @ self.z).reshape(-1,self.dim)
            if name == 'mesh':
                print("Clicked object : ", name)
                is_vertex = i < P.shape[0]
                is_face = i > P.shape[0] and (i < (self.X.shape[0] + self.T.shape[0]))
                
                if is_vertex:
                    print("Clicked vertex index : ", i)
                    self.bc = np.array([P[ i, :]])
                    self.bI = np.array([i])
                elif is_face:
                    print("Clicked face index : ", i - P.shape[0])
                    Faces = igl.boundary_facets(self.T)
                    self.bc = np.array([P[Faces[i-P.shape[0], 0]]])
                    self.bI = np.array([Faces[i - P.shape[0], 0]])

                if is_vertex or is_face:
                    Q_picked, b_picked, self.SGamma = dirichlet_penalty(self.bI, self.bc, P.shape[0],  self.params.k_picked, only_b=False, return_SGamma=True)
                    self.Q_picked = self.sim.B.T @ Q_picked @ self.sim.B

                    self.b_picked = self.sim.B.T @ b_picked

                    self.sim.Q = self.BQB_ext + self.Q_picked
                    self.sim.recompute_system()
                    # b = -self.BSGamma @ self.bc.reshape(-1, 1)
                    self.BSGamma = self.sim.B.T @ self.SGamma 
                    self.picked_pc = ps.register_point_cloud("picked", self.bc.reshape(1, -1) )
                    self.picked_pc.add_scalar_quantity("rad", np.array([0.05]))
                    self.picked_pc.set_point_radius_quantity("rad", autoscale=False)

    
        if self.bc is not None:
            if psim.IsKeyPressed(psim.ImGuiKey_J):
                self.bc += self.speed * np.array([[-1, 0., 0]])
            if psim.IsKeyPressed(psim.ImGuiKey_L):
                self.bc += self.speed * np.array([[1, 0., 0]])
            if psim.IsKeyPressed(psim.ImGuiKey_I):
                self.bc += self.speed * np.array([[0, 1., 0]])
            if psim.IsKeyPressed(psim.ImGuiKey_K):
                self.bc += self.speed * np.array([[0, -1., 0]])
            if psim.IsKeyPressed(psim.ImGuiKey_U):
                self.bc += self.speed * np.array([[0, 0., 1]])
            if psim.IsKeyPressed(psim.ImGuiKey_O): 
                self.bc += self.speed * np.array([[0, 0., -1]])

            self.b_picked = -self.BSGamma @ self.bc.reshape(-1, 1)

            if psim.IsKeyPressed(psim.ImGuiKey_R):
                self.recording = not self.recording

                if not self.recording :
                    os.makedirs(self.recording_dir, exist_ok=True)
                    print("Saving recording to : ", self.recording_dir)

                    
        if not self.pause:
            for _i in range(self.params.substeps):

                z_next = self.sim.step(self.z,  self.z_dot, b_ext=self.Bb_ext + self.b_picked)
                
                self.z_dot = (z_next - self.z) / self.sim.params.h    
                self.z = z_next.copy()

            
            # update geometry
            x = self.sub.B @ self.z 
            U = x.reshape(-1, self.dim)


            # update visuals 
            if self.use_tex:
                self.mesh.update_vertex_positions(U[self.faceI, :])
            else:
                self.mesh.update_vertex_positions(U)

            # self.points.update_point_positions(U[self.sub.conI, :])

            if self.bI is not None:
                self.picked_pc.update_point_positions((self.bc).reshape(1, -1))

        end_time = time.time()
        total_time = end_time - start_time
        self.timings[self.counter % self.timings.shape[0]] = total_time
        self.counter += 1
        
        psim.Text("Time taken for sim step : " + str(self.timings.mean()))
        psim.Text("FPS : " + str(1/self.timings.mean()))

