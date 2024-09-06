import igl
import numpy as np
import scipy as sp
from simkit import deformation_jacobian, project_into_subspace, quadratic_hessian, selection_matrix

from ...solvers import NewtonSolver, NewtonSolverParams
from ... import ympr_to_lame
from ... import elastic_energy_S, elastic_gradient_dS, elastic_hessian_d2S
from ... import quadratic_energy, quadratic_gradient, quadratic_hessian
from ... import kinetic_energy_z, kinetic_gradient_z, kinetic_hessian_z, KineticEnergyZPrecomp
from ... import stretch, stretch_gradient_dz
from ... import volume
from ... import massmatrix
from ... import backtracking_line_search

from ...project_into_subspace import project_into_subspace
from ...symmetric_stretch_map import symmetric_stretch_map

from ..Sim import  *
from ..State import State

from ...solvers.SQPMFEMSolver import SQPMFEMSolver, SQPMFEMSolverParams

class ElasticROMMFEMState(State):

    def __init__(self, z, c, l, z_dot):
        self.z = z
        self.c = c
        self.l = l
        self.z_dot = z_dot
        return


class ElasticROMMFEMSimParams():
    def __init__(self, rho=1, h=1e-2, ym=1, pr=0,  material='arap', gamma=1,
                 solver_p : SQPMFEMSolverParams  = None, Q0=None, b0=None):
        """
        Parameters of the pinned pendulum simulation

        Parameters
        ----------

        """
        self.h = h
        self.rho = rho
        self.ym = ym
        self.pr = pr
        self.material = material

        if solver_p is None:
            solver_p = NewtonSolverParams() #SQPMFEMSolverParams()
        self.solver_p = solver_p
        self.Q0 = Q0
        self.b0 = b0
        self.gamma = gamma
        
        return


class ElasticROMMFEMSim(Sim):

    def __init__(self, X : np.ndarray, T : np.ndarray, B : np.ndarray, cI=None,  cW=None, p : ElasticROMMFEMSimParams = ElasticROMMFEMSimParams()):
        """

        Parameters
        ----------
        p : ElasticFEMSimParams
            Parameters of the elastic system
        """
        
        dim = X.shape[1]
        self.dim = dim

        self.p = p
        # preprocess some quantities
        self.X = X
        self.T = T

        # i hate initializing state here. get rid of this.
        x = X.reshape(-1, 1)


        self.B = B
     
        ## cubature precomp
        if cI is not None:
            assert cW is not None
            assert cW.shape[0] == cI.shape[0]
        if cI is None:
            cI = np.arange(0, T.shape[0])

        self.cI = cI
        [self.kin_pre, self.GJB, self.mu, self.lam, self.vol, self.C, self.Ci] =  \
            self.initial_precomp(X, T, B, cI, cW, p.rho, p.ym, p.pr, dim)

        self.nz = B.shape[-1]
        self.na = self.Ci.shape[0]
    
        # should also build the solver parameters
        if isinstance(p.solver_p, NewtonSolverParams):
            self.solver = NewtonSolver(self.energy, self.gradient, self.hessian, p.solver_p)
        elif isinstance(p.solver_p, SQPMFEMSolverParams):
            self.solver = SQPMFEMSolver(self.energy, self.hessian_blocks, self.gradient_blocks, p.solver_p)
        else:
            # print error message and terminate programme
            assert(False, "Error: solver_p of type " + str(type(p.solver_p)) +
                          " is not a valid instance of NewtonSolverParams. Exiting.")
        return

    def initial_precomp(self, X, T, B,  cI, cW, rho, ym, pr, dim):
        
        # kinetic energy precomp
        M = massmatrix(self.X, self.T, rho=rho)
        Mv = sp.sparse.kron( M, sp.sparse.identity(dim))# sp.sparse.block_diag([M for i in range(dim)])
        kin_z_precomp = KineticEnergyZPrecomp(B, Mv)
        
        
        # elastic energy precomp
        if cW is None:
            vol = volume(X, T)
        else:
            vol = cW.reshape(-1, 1)
            
        ## ym, pr to lame parameters
        mu, lam = ympr_to_lame(ym, pr)
        if isinstance(mu, float):
            mu = np.ones((T.shape[0], 1)) * mu
        if isinstance(lam, float):
            lam = np.ones((T.shape[0], 1)) * lam

        mu = mu[cI]
        lam = lam[cI]

        ## selection matrix from cubature precomp.
        G = selection_matrix(cI, T.shape[0])
        Ge = sp.sparse.kron(G, sp.sparse.identity(dim*dim))
        J = deformation_jacobian(self.X, self.T)
        GJB = Ge @ J @ B
        # elastic_z_precomp = ElasticEnergyZPrecomp(B, Ge, J, self.dim)
        
        C, Ci = symmetric_stretch_map(cI.shape[0], dim)

   
        return  kin_z_precomp, GJB, mu, lam, vol, C, Ci

    def dynamic_precomp(self, z, z_dot, Q_ext=None, b_ext=None):
        """
        Computation done once every timestep and never again
        """
        self.y = z + self.p.h * z_dot

        # add to current Q_ext
        if Q_ext is not None:
            if self.p.Q0 is None:
                self.Q = Q_ext
            else:
                self.Q = Q_ext + self.p.Q0

        # same for b
        if b_ext is not None:
            if self.p.b0 is None:
                self.b = b_ext
            else:
                self.b = b_ext + self.p.b0

                
    def energy(self, p : np.ndarray):
        dim = self.dim

        z, s = self.z_a_from_p(p)

        S = s.reshape(-1, dim * (dim + 1) // 2)
        F = (self.GJB @ z).reshape(-1, dim, dim) # deformation gradient at cubature tets
        
        elastic = elastic_energy_S(S, self.mu, self.lam,  self.vol, self.p.material)

        quad =  quadratic_energy(z, self.Q, self.b)

        kinetic = kinetic_energy_z(z, self.y, self.p.h, self.kin_pre)

        constraint = (np.linalg.norm(stretch(F) - self.C @ s)) * self.p.gamma # merit function

        total = elastic + quad + kinetic + constraint
        return total


    def gradient(self, p : np.ndarray):
        dim = self.dim

        z, a = self.z_a_from_p(p)

        g_z, g_s, g_l = self.gradient_blocks(p)
    
        g = np.vstack([g_z, g_s, g_l])
        return g


    def hessian(self, p : np.ndarray):
        dim = self.dim
        z, a = self.z_a_from_p(p)

        # F = (self.GJ @ z).reshape(-1, dim, dim)
        A = a.reshape(-1, dim * (dim + 1) // 2)

        H_zz, H_ss, H_zl, H_sl, H_sli = self.hessian_blocks(p)
        
        H_zs = sp.sparse.csc_matrix((z.shape[0], a.shape[0])) 
        H_ll =  sp.sparse.csc_matrix((a.shape[0], a.shape[0])) 

        H = sp.sparse.bmat([[H_zz,    H_zs,   H_zl], 
                            [H_zs.T,  H_ss,   H_sl], 
                            [H_zl.T,  H_sl, H_ll]])
        # H = H.toarray()
        return H
    

    def hessian_blocks(self, p : np.ndarray):
        dim = self.dim
        z, a = self.z_a_from_p(p)
        # F = (self.GJ @ z).reshape(-1, dim, dim)
        A = a.reshape(-1, dim * (dim + 1) // 2)

        H_zz = kinetic_hessian_z(self.p.h, self.kin_pre)+ \
            quadratic_hessian(self.Q)
        
        H_zl = stretch_gradient_dz(z, self.GJB, Ci=self.Ci, dim=dim) 

        H_ss =  elastic_hessian_d2S(A, self.mu, self.lam, self.vol, self.p.material) 
    
        H_sl = -sp.sparse.identity(a.shape[0])
        H_sli = H_sl.copy()

        return H_zz, H_ss, H_zl, H_sl, H_sli
    
    def gradient_blocks(self, p : np.ndarray):
        dim = self.dim
        z, a = self.z_a_from_p(p)
        F = (self.GJB @ z).reshape(-1, dim, dim)
        A = a.reshape(-1, dim * (dim + 1) // 2)
        g_x = kinetic_gradient_z(z, self.y, self.p.h, self.kin_pre) \
                + quadratic_gradient(z, self.Q, self.b) 
        g_s =  elastic_gradient_dS(A,  self.mu, self.lam, self.vol, self.p.material)
        g_mu = (self.Ci @ stretch(F) - a)
        return g_x, g_s, g_mu
    
    def step(self, z : np.ndarray, s : np.ndarray,  z_dot : np.ndarray, Q_ext=None, b_ext=None):
    
        # call this to set up inertia forces that change each timestep.
        self.dynamic_precomp(z, z_dot, Q_ext, b_ext)

        l = np.zeros(s.shape)
        p = np.vstack([z, s, l])

        p = self.solver.solve(p)
        
        z, s = self.z_a_from_p(p)
        return z, s


    def z_a_from_p( self, p):
        z = p[:self.nz]
        a = p[self.nz:self.nz + self.na]
        return z, a
    
    def p_from_z_c(self, z, a):
        return np.concatenate([z, a])



# array([[-0.5       ],
#        [ 0.07458123],
#        [ 0.5       ],
#        [ 0.07458123],
#        [ 0.        ],
#        [ 0.07458123],
#        [ 0.5       ],
#        [ 0.03729061],
#        [-0.25      ],
#        [ 0.07458123],
#        [-0.5       ],
#        [ 0.03729061],
#        [ 0.25      ],
#        [ 0.07458123],
#        [ 0.        ],
#        [ 0.03729061],
#        [-0.25      ],
#        [ 0.03729061],
#        [ 0.25      ],
#        [ 0.03729061],
#        [ 0.5       ],
#        [ 0.05593592],
#        [-0.375     ],
#        [ 0.07458123],
#        [-0.5       ],
#        [ 0.01864531],
#        [ 0.125     ],
#        [ 0.07458123],
#        [ 0.        ],
#        [ 0.05593592],
#        [ 0.5       ],
#        [ 0.01864531],
#        [-0.125     ],
#        [ 0.07458123],
#        [-0.5       ],
#        [ 0.05593592],
#        [ 0.375     ],
#        [ 0.07458123],
#        [ 0.        ],
#        [ 0.01864531],
#        [-0.25      ],
#        [ 0.05593592],
#        [-0.25      ],
#        [ 0.01864531],
#        [-0.375     ],
#        [ 0.03729061],
#        [-0.125     ],
#        [ 0.03729061],
#        [ 0.25      ],
#        [ 0.05593592],
#        [ 0.25      ],
#        [ 0.01864531],
#        [ 0.125     ],
#        [ 0.03729061],
#        [ 0.375     ],
#        [ 0.03729061],
#        [ 0.375     ],
#        [ 0.01864531],
#        [ 0.125     ],
#        [ 0.01864531],
#        [ 0.125     ],
#        [ 0.05593592],
#        [-0.125     ],
#        [ 0.01864531],
#        [-0.375     ],
#        [ 0.01864531],
#        [-0.375     ],
#        [ 0.05593592],
#        [-0.125     ],
#        [ 0.05593592],
#        [ 0.375     ],
#        [ 0.05593592]]





# array([[ 0.        ],
#        [ 5.71012515],
#        [ 0.        ],
#        [ 5.71012515],
#        [ 0.        ],
#        [11.42025029],
#        [ 0.        ],
#        [11.42025029],
#        [ 0.        ],
#        [11.42025029],
#        [ 0.        ],
#        [11.42025029],
#        [ 0.        ],
#        [11.42025029],
#        [ 0.        ],
#        [22.84050059],
#        [ 0.        ],
#        [22.84050059],
#        [ 0.        ],
#        [22.84050059],
#        [ 0.        ],
#        [11.42025029],
#        [ 0.        ],
#        [11.42025029],
#        [ 0.        ],
#        [ 5.71012515],
#        [ 0.        ],
#        [11.42025029],
#        [ 0.        ],
#        [22.84050059],
#        [ 0.        ],
#        [ 5.71012515],
#        [ 0.        ],
#        [11.42025029],
#        [ 0.        ],
#        [11.42025029],
#        [ 0.        ],
#        [11.42025029],
#        [ 0.        ],
#        [11.42025029],
#        [ 0.        ],
#        [22.84050059],
#        [ 0.        ],
#        [11.42025029],
#        [ 0.        ],
#        [22.84050059],
#        [ 0.        ],
#        [22.84050059],
#        [ 0.        ],
#        [22.84050059],
#        [ 0.        ],
#        [11.42025029],
#        [ 0.        ],
#        [22.84050059],
#        [ 0.        ],
#        [22.84050059],
#        [ 0.        ],
#        [11.42025029],
#        [ 0.        ],
#        [11.42025029],
#        [ 0.        ],
#        [22.84050059],
#        [ 0.        ],
#        [11.42025029],
#        [ 0.        ],
#        [11.42025029],
#        [ 0.        ],
#        [22.84050059],
#        [ 0.        ],
#        [22.84050059],
#        [ 0.        ],
#        [22.84050059],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ],
#        [ 0.        ]])