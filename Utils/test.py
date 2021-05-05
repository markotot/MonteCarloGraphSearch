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

graph = nx.fast_gnp_random_graph(1000, 0.05, seed=None, directed=True)

pos = graphviz_layout(graph, prog='neato')


for node in graph.nodes:
    for node_2 in graph.nodes:
        nx.has_path(graph, node, node_2)
