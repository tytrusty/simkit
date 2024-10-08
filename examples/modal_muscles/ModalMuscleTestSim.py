
import igl
import numpy as np
import scipy as sp
import os
from copy import deepcopy

from simkit.ympr_to_lame import ympr_to_lame
from simkit.deformation_jacobian import deformation_jacobian
from simkit.cluster_grouping_matrices import cluster_grouping_matrices
from simkit.polar_svd import polar_svd
from simkit.volume import volume
from simkit.massmatrix import massmatrix
from process_delta import process_delta
from simkit.gravity_force import gravity_force
from process_tet_field import process_tet_field
# import time
import timeit



class ModalMuscleTestSimParams():
     def __init__(s, dt=1e-2, rho=1e3, mu=0,
                  g=-10, iters=10,ground_contact=False, ground_height=0,
                  contact_reg = 1e-3, muc=1, mud=0.9,
                  contact_method='projective_velocity_implicit', threshold=1e-6, K=None, BKB=None):
        s.dt = dt
        s.rho = rho
        s.mu = mu
        s.g = g
        s.iters = iters
        s.ground_contact = ground_contact
        s.ground_height = ground_height
        s.contact_reg = contact_reg
        s.muc = muc
        s.mud = mud
        s.contact_method = contact_method
        s.threshold = threshold


        s.K = K
        s.BKB = BKB


     def process_mu(s, T):
        return process_tet_field(s.mu, T)

     def set_K(s, K):
        s.K = K

     def set_BKB(s, BKB):
        s.BKB = BKB



class ModalMuscleTestSim():
    def __init__(s, X, T, B, l, cI, p : ModalMuscleTestSimParams = ModalMuscleTestSimParams()):
        mu = p.process_mu(T)

        s.X = X
        s.T = T
        s.dt = p.dt
        s.dt2 = p.dt * p.dt
        s.n = X.shape[0]
        s.t = T.shape[0]

        s.k = l.max() + 1
        s.m = B.shape[1]
        s.mm = s.m

        s.dim = X.shape[1]
        dim = s.dim
        s.max_iters = p.iters


        s.B = B
        Ms =massmatrix(X, T)

        Ms2 = massmatrix(X, T, rho=2)
        M = sp.sparse.kron( massmatrix(X, T, rho=p.rho), sp.sparse.identity(s.dim))

        s.rho = p.rho;

        [G, Gm, mc, mt, f] = cluster_grouping_matrices(l, X, T, return_mass=True)

        Ge = sp.sparse.kron(G, sp.sparse.identity(dim*dim))
        Gme = sp.sparse.kron(Gm, sp.sparse.identity((dim*dim)))
        s.f = f

        if (p.ground_height is None):
            s.ground_h = 0
        else:
            s.ground_h = p.ground_height

        s.ground_contact = p.ground_contact
        s.muc = p.muc
        s.mud = p.mud

        ## Precompute static matrices
        J = deformation_jacobian(X, T)
        vol = volume(X, T)[:, 0]

        A =  sp.sparse.kron(sp.sparse.diags(vol), sp.sparse.identity(dim*dim))
        JA = J.T @ A
        s.mu = mu
        muv = (np.kron(mu, np.ones(dim*dim)))# * 0 + 1e2
        Mu = sp.sparse.diags(muv)

        L = JA  @ Mu @ J
        x0 = X.reshape(-1, 1)
        Jx0 = J @ x0
        Lx0 = L @ x0
        Q =  L + M / s.dt2

        # JB = J @ B
        GmeJ = Gme @ J
        s.GmJB = (GmeJ) @ B
        s.GmJx0 = Gme  @ Jx0
        s.BLx0 = B.T @ Lx0

        JAMuGe = JA @ Mu @ Ge.T

        s.BJAMuG = B.T @ JAMuGe
        s.SMuAJx0 = JAMuGe.T @ x0

        BM = (B.T @ M) 
        s.BMB = (BM @ B) 
        s.BLB = (B.T @ L @ B)
        s.BQB = (B.T @ Q @ B)

        s.BKB = np.zeros(s.BMB.shape)
        if p.BKB is not None:
            s.BKB = p.BKB
        if p.K is not None: # add qua)
            s.BKB += (B.T @ p.K @ B)#sp.sparse.csc_matrix((s.n*3, s.n*3))

        s.BQB += s.dt2*s.BKB
        s.is_sparse = False
        if (sp.sparse.issparse(B)):
            s.is_sparse = True
            s.solve_BQB = sp.sparse.linalg.factorized(s.BQB)
        else:
            s.chol_BQB = sp.linalg.cho_factor(s.BQB)

        # gg = np.array([[0, p.g, 0]])[0, :dim]
        gg = np.array([[0, p.g, 0]])[0, :dim]
        fg = np.tile(gg, (s.n, 1)).reshape(-1, 1)
        s.g =   BM @ fg

        s.threshold = p.threshold

        x0 = X.reshape(-1, 1)

        s.contact_method = p.contact_method
        if s.ground_contact:
            s.cI = cI
            s.sn = s.cI.shape[0]
            S = sp.sparse.csc_matrix((np.ones(s.sn), (np.arange(s.sn), cI)), shape=(s.sn, s.n))
            Se = sp.sparse.kron( S, sp.sparse.identity(dim))

            # contact jacobian, maps from reduced displacement space to contact point displacements
            s.D = Se @ B

            # contact point rest positions
            s.Sx0 = Se @ x0

            if s.is_sparse:
                DQi = sp.sparse.linalg.spsolve(s.BQB, s.D.T).T
            else:
                DQi = sp.linalg.cho_solve(s.chol_BQB, s.D.T).T

            s.DQi = DQi
            # # only useful for implicit
            # if s.contact_method == "projective_velocity_implicit" or \
            #     s.contact_method == "projective_velocity_implicit_mass":
            #     s.QiDDQi = s.DQi.T @ s.DQi
            #     s.DQiQiD = s.DQi @ s.DQi.T
            #     if s.DQi.shape[0] <= s.DQi.shape[1]:
            #         s.underconstrained = True
            #         if s.is_sparse:
            #             chol_DQiQiD = sp.sparse.linalg.factorized(s.DQiQiD)
            #         else:
            #             chol_DQiQiD = sp.linalg.cho_factor(s.DQiQiD)
            #         s.chol_DQiQiD = chol_DQiQiD


    def step(s, zn_curr, zn_vel):
        g = s.g
        zn1 = zn_curr
        zn1_vel = zn_vel

        y =  zn_curr + s.dt * zn_vel 

        res = np.inf

        for i in range(0, s.max_iters):
            res_prev = deepcopy(res)

            zn1_prev = deepcopy(zn1)

            f_l_ext = np.zeros((s.mm, 1))
            r = s.local_step(zn1)

            b_arap = s.BJAMuG @ r - s.BLx0

            b =  s.BMB @ y /s.dt2 +  g  +  b_arap +  f_l_ext

            b_tilde = (b - s.BQB @ zn_curr) / s.dt

            if s.ground_contact:
                c = s.ground_contact_projective_velocity_implicit(zn1, zn1_vel, b_tilde)
            else:
                c = np.zeros(b.shape)
            
            b_tilde += c

            zn1_vel = s.global_step(b_tilde)

            zn1 = zn_curr + s.dt * zn1_vel

            res = np.linalg.norm(zn1 - zn1_prev)

            if res < s.threshold or res == res_prev:
                break
        return zn1, zn1_vel


    def local_step(s, z):
        # U = u.reshape((s.dim, s.n,)).T
        f = (s.GmJB @ z) + s.GmJx0
        f = f.reshape((-1, s.dim,  s.dim,))
        #
        # [R0, S0] = polar_svd(f, flip=True)

        [R, S] =  polar_svd(f, flip=True)
        # R = R.transpose(0, 2, 1)
        r = R.reshape((-1, 1))
        return r

    def global_step(s, b):
        if s.is_sparse:
            z = s.solve_BQB(b)
        else:
            z = sp.linalg.cho_solve( s.chol_BQB, b)

        return z

    def ground_contact_projective_velocity_implicit(s,zn1, zn1_vel, b):
        # this is a velocity now
        z_vel_tent = sp.linalg.cho_solve(s.chol_BQB, b)
        # project to contact points
        p = s.D @ (zn1) + s.Sx0

        P = p.reshape((-1, s.dim))
        C = (s.D @ z_vel_tent).reshape((p.shape[0] // s.dim, s.dim))

        thresh = 0
        cI = (P[:, 1] < (s.ground_h))
        nI = (C[:, 1] < (0)) * cI  # interpenetrating

        nI = np.where(nI)[0]

        sn = nI.shape[0]
        c = np.zeros(zn1.shape)
        if sn > 0:
            #full static friction
            if s.muc == 1:
                if s.dim== 3:
                    i = np.array([nI, nI + s.sn, nI + 2 * s.sn]).reshape(-1)
                elif s.dim == 2:
                    i = np.array([nI, nI + s.sn]).reshape(-1)
                DQi = s.DQi[i, :]#.squeeze()
            else:
            #no friction
                i = np.array([nI + s.sn]).reshape(-1)
                DQi = s.DQi[i, :]

            if s.muc == 1:
                rhs = np.zeros((sn,s.dim))
                vel_tent = (DQi @ b).reshape(-1, s.dim)
                if s.dim == 3:
                    rhs[:, [0, 2]] = (1 - s.mud) * vel_tent[:, [0, 2]]
                elif s.dim == 2:
                    rhs[:, [0]] = (1 - s.mud) * vel_tent[:, [0]]

                # pos_contacting_points = P[nI, :]
                # closest_ground_point = pos_contacting_points.copy()
                # closest_ground_point[:, [1]] = s.ground_h
                # target_normal_vel = (closest_ground_point - pos_contacting_points)*s.dt
                # rhs[:, [1]] = target_normal_vel[:, [1]]
            else:
                rhs = np.zeros((sn,1))
                # pos_contacting_points = P[nI, :]
                # closest_ground_point = pos_contacting_points.copy()
                # closest_ground_point[:, [1]] = s.ground_h
                # target_normal_vel = closest_ground_point - pos_contacting_points
                # rhs = target_normal_vel[:, [1]]
            # figure out rhs for all interpenetrating rhs

            rhs = rhs.reshape(-1, 1)

            if DQi.shape[0] >= DQi.shape[1]:
                QiDDQi = DQi.T @ DQi
                # overconstrained, do least squares
                c = np.linalg.solve(QiDDQi, DQi.T @ (rhs- DQi @ b)) # @ z_tent))
                # f = (D @ c).reshape((-1, 1), order='F')
            else:
                DQiQiD = DQi @ DQi.T
                # force required to stop interpenetration, aka Normal Force
                c = DQi.T @ np.linalg.solve(DQiQiD, (rhs - DQi @ b )) # z_tent))

        # import polyscope as ps


            # ps.init()
            # ft = (Dt @ b)
        # fc = (s.D @ c).reshape(-1, 2, order='F')

        # f = (s.D @ (b+c)).reshape(-1, 2, order='F')
        # ps.get_point_cloud('cI').add_vector_quantity('f', 0.01 * (f), enabled=True, vectortype='ambient',
        #                                              radius=0.05)
        # ps.get_point_cloud('cI').add_vector_quantity('fc', 0.0001 * (fc), enabled=True, vectortype='ambient',
        #                                              radius=0.05)
        # ps.frame_tick()

        return c



    def arap_energy_reduced(s, z):

        r = s.local_step(z)

        E = 0.5 * z.T @ s.BLB @ z + \
                 z.T @ (s.BLx0 - s.BJAMuG @ r) \
                 - s.SMuAJx0.T @ r

        return E

    def kinetic_energy_reduced(self, z, z_vel):
        # y = z_prev + self.dt * z_vel
        # d = z - y
        E = 0.5 * ( z_vel.T @ self.BMB @ z_vel) / self.dt2
        return E


    def contact_energy(self, z, z_vel, f_ext=None):
        return 0


    def energy_reduced(s, z, z_vel, z_prev):
        print("Energy Redcued Doesn't Take into Account Contact Yet")
        E = s.arap_energy_reduced(z) + s.kinetic_energy_reduced(z, z_vel, z_prev)  + s.contact_energy(z, z_vel)
        return E


    def gradient_reduced(s, z, z_vel, zn_curr, zn_vel_curr, f_ext=None, f_local_ext=None):
        g = s.g
        zn1 = zn_curr
        zn1_vel = zn_vel_curr
        y =  zn_curr + s.dt * zn_vel_curr + s.dt2 * g
        for i in range(0, s.max_iters):

            zn1_prev = zn1.copy()

            f_l_ext = np.zeros((s.mm, 1))
            if f_local_ext is not None:
                f_l_ext = f_local_ext(zn1, zn1_vel)

            r = s.local_step(zn1)
            b_arap = s.BJAMuG @ r - s.BLx0
            b = (s.rho * y + s.dt2 * f_ext) + s.dt2 * b_arap + s.dt2 * f_l_ext

            g = s.Q @ z - b

        return g

    # def render(s,  X, T, B, sim_sub_hist, cI=None, eye_pos=None, eye_target=None, render_dir=None, l=None, mesh_dir=None):
    #     ps.init()
    #     ps.remove_all_structures()
    #     num_controller_steps = sim_sub_hist['z'].shape[0]
    #     num_sim_substeps = sim_sub_hist['z'].shape[1]
    #     Z = sim_sub_hist['z']
    #     u = B @ Z
    # 
    #     if s.ground_contact:
    #         contact_points = ps.register_point_cloud("contact", X[cI, :])
    #         contact_points.set_color([0, 0, 0])
    #         ps.set_ground_plane_mode("tile_reflection")
    #     else:
    #         ps.set_ground_plane_mode("none")
    # 
    #     ps.set_bounding_box([-1, -0, -1], [1, 1, 1])
    # 
    # 
    # 
    #     counter = 0
    #     controller_step = 0
    #     done = False
    #     if render_dir is not None:
    #         step_dir = render_dir + "/controller_step_0000/"
    #         os.makedirs(step_dir, exist_ok=True)
    # 
    #     if mesh_dir is not None:
    #         os.makedirs(mesh_dir, exist_ok=True)
    #     def sim_playback_callback():
    #         nonlocal counter, controller_step, step_dir, done
    # 
    #         if not done:
    #             if render_dir is not None:
    #                 ps.screenshot(step_dir + "/" + str(counter).zfill(4) + ".png", transparent_bg=False)
    # 
    #             U = np.reshape(u[controller_step, counter], (-1, 3), order='F')
    #             P = U + X
    #             mesh.update_vertex_positions(P)
    #             if s.ground_contact:
    #                 contact_points.update_point_positions(P[cI, :])
    # 
    #             if mesh_dir is not None:
    #                 igl.write_obj(mesh_dir + "/" + str(counter).zfill(4) + ".obj", P, igl.boundary_facets(T))
    # 
    #             counter += 1
    #             if counter == num_sim_substeps:
    #                 counter = 0
    #                 if render_dir is not None:
    #                     video_from_image_dir(step_dir, step_dir + "/../controller_step_" + str(controller_step).zfill(4) + ".mp4")
    #                 controller_step += 1
    #                 if controller_step == num_controller_steps:
    #                     done = True
    #                     return
    #                 else:
    #                     if render_dir is not None:
    #                         step_dir = render_dir + "/controller_step_" + str(controller_step).zfill(4) + "/"
    #                         os.makedirs(step_dir, exist_ok=True)
    #         else:
    #             os.system("taskkill /fi \"WINDOWTITLE eq Polyscope\"")
    #             return
    # 
    # 
    #     mesh = ps.register_volume_mesh('ps_mesh', X, T)
    #     mesh.set_color(np.array([49, 130, 189]) / 255)
    # 
    #     if l is not None:
    #         mesh.add_scalar_quantity('clusters', l, enabled=False, defined_on='cells', cmap='rainbow')
    #     if s.mu is not None:
    #         mesh.add_scalar_quantity("mu", s.mu, enabled=True, defined_on='cells')
    # 
    #     if eye_pos is not None and eye_target is not None:
    #         ps.look_at(eye_pos, eye_target)
    # 
    #     # Z['z'].reshape()
    #     ps.set_give_focus_on_show(True)
    #     ps.set_user_callback(sim_playback_callback)
    # 
    #     ps.show()
    #     ps.show()
    #     ps.show()
    # 
    #     if render_dir is not None:
    #         concatenate_videos(render_dir, render_dir + "/combined.mp4")
    #         mp4_to_gif( render_dir + "/combined.mp4", render_dir + "/combined.gif")
    # 
    #     return
    # 

    def rest_state(self):

        z = np.zeros((self.BQB.shape[0], 1))
        z_vel = np.zeros((self.BQB.shape[0], 1))
        return z, z_vel