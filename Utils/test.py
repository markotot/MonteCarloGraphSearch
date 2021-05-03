import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from Utils.Logger import Logger
from networkx.drawing.nx_agraph import graphviz_layout
from Agents.MCGS.StateDatabase import StateDatabase
from Agents.MCGS.Graph import Graph
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

path = "../graph.gpickle"
graph = Graph()
graph.load_graph(path)
print("Oy")
pos = graphviz_layout(graph.graph, prog='dot')


dpi = 96
plt.figure(1, figsize=(1024/dpi, 768/dpi))
print("1y")
nx.draw(graph.graph, pos)
print("2y")
plt.show()