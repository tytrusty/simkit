import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from matplotlib.lines import Line2D
class Curve():
    def __init__(self, X, E, color=None, linestyle='solid', linewidth=2, label="curve"):
        self.X = X
        self.E = E

        lines = [(X[edge[0]], X[edge[1]]) for edge in E]
        self.lc = LineCollection(lines, color=color , linestyle=linestyle, linewidth=linewidth)
        plt.gca().add_collection(self.lc)
        
        self.proxy_line = Line2D([0], [0], color=color, linewidth=linewidth, linestyle=linestyle, label=label)

    def update_vertex_positions(self, X):
        self.X = X
        lines = [(X[edge[0]], X[edge[1]]) for edge in self.E]
        self.lc.set_segments(lines)

    def remove(self):
        self.lc.remove()
        del self.lc