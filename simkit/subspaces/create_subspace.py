
from numpy import isin
from .Subspace import SubspaceParams

from .SkinningEigenmodes import SkinningEigenmodesParams, SkinningEigenmodes
from .SpectralLocalizedSkinningEigenmodes import SpectralLocalizedSkinningEigenmodesParams, SpectralLocalizedSkinningEigenmode
from .Full import FullParams, Full
from .ModalAnalysis import ModalAnalysisParams, ModalAnalysis
from .RandomImpulseVibes import RandomImpulseVibesParams, RandomImpulseVibes

from .GaussianRBF import GaussianRBFParams, GaussianRBF

def create_subspace(X, T, p : SubspaceParams):

    if isinstance(p, SkinningEigenmodesParams):
        return SkinningEigenmodes(X, T, p)
    if isinstance(p, ModalAnalysisParams):
        return ModalAnalysis(X, T, p)
    elif isinstance(p, SpectralLocalizedSkinningEigenmodesParams):
        return SpectralLocalizedSkinningEigenmode(X, T, p)
    elif isinstance(p, GaussianRBFParams):
        return GaussianRBF(X, T, p)
    elif isinstance(p, FullParams):
        return Full(X, T, p)
    elif isinstance(p, RandomImpulseVibesParams):
        return RandomImpulseVibes(X, T, p)
    else:
        NotImplementedError("Subspace type not implemented")
        return 