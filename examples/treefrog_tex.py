import numpy as np
import polyscope as ps
import igl


[V, UV, _, F, FUV, _ ] = igl.read_obj("../data/3d/treefrog/texture.obj")
ps.init()


UV = UV[:V.shape[0], :]

ps_mesh = ps.register_surface_mesh("my mesh", V, F)

# ps.show()

# add a parameterization (aka UV map)
# param_vals = np.random.rand(ps_mesh.n_vertices(), 2)
ps_mesh.add_parameterization_quantity("test_param", UV,
                                      defined_on='vertices', enabled=True)

# add the texture quantity
# dims = (200,300)
# vals = np.random.rand(*dims) # dummy placeholder image data

from PIL import Image
import numpy as np
vals = np.array(Image.open("../data/3d/treefrog/texture.png"))
ps_mesh.add_scalar_quantity("test_vals", vals,
                             defined_on='texture', param_name="test_param",
                             vminmax=(-5., 5.), enabled=True)

ps.show()