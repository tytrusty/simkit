
import polyscope as ps
import igl
from PIL import Image
import numpy as np
[V, UV, _, F, FUV, _ ] = igl.read_obj("../data/3d/treefrog/treefrog_tex.obj")
ps.init()
ps_mesh = ps.register_surface_mesh("treefrog", V, F)
fuv = FUV.flatten()
uv = UV[fuv, :]
ps_mesh.add_parameterization_quantity("test_param",  uv,
                                      defined_on='corners', enabled=True)
vals = np.array(Image.open("../data/3d/treefrog/treefrog_tex.png"))
ps_mesh.add_color_quantity("test_vals", vals[:, :, 0:3]/255,
                             defined_on='texture', param_name="test_param",
                              enabled=True)
ps.show()