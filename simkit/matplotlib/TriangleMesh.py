
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from .colors import *
class TriangleMesh():
    def __init__(self, X, T, facecolors=light_blue, edgecolors=black, linewidths=1, label="mesh"):
        self.X = X
        self.T = T

        triangles = [X[face] for face in T]

        self.pc = PolyCollection(triangles, facecolors=facecolors, edgecolors=edgecolors, linewidths=linewidths)
        plt.gca().add_collection(self.pc)
        
        return

    def update_vertex_positions(self, X):
        self.X = X
        triangles = [X[face] for face in self.T]
        self.pc.set_verts(triangles)
        return

    def remove(self):
        self.pc.remove()
        return