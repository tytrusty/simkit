from .ElasticFEMSim import ElasticFEMSim, ElasticFEMSimParams
from .ElasticROMFEMSim import ElasticROMFEMSim, ElasticROMFEMSimParams

from .ElasticMFEMSim import ElasticMFEMSim, ElasticMFEMSimParams
from .ElasticROMMFEMSim import ElasticROMMFEMSim, ElasticROMMFEMSimParams



def create_elastic_sim(X, T, sim_params, sub=None):

    if isinstance(sim_params, ElasticFEMSimParams):
        return ElasticFEMSim(X, T, sim_params)
    elif isinstance(sim_params, ElasticROMFEMSimParams):
        assert sub is not None
        return ElasticROMFEMSim(X, T, sub.B, cI=sub.cI, cW=sub.cW, p=sim_params)
    elif isinstance(sim_params, ElasticMFEMSimParams):
        return ElasticMFEMSim(X, T, sim_params)
    elif isinstance(sim_params, ElasticROMMFEMSimParams):
        assert(sub is not None)
        return ElasticROMMFEMSim(X, T, sub.B, cI=sub.cI, cW=sub.cW, p=sim_params)
    else:
        raise ValueError("Unknown sim_params type")
    return

