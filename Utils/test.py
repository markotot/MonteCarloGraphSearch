import concurrent.futures
import math
from Agents.MCGS.Graph import Graph
import networkx as nx
import numpy as np
seed = 42
config = {
    'amplitude_factor': 0.1,
    'noisy_min_value': 0.1,
}

graph = Graph(seed, config)
graph.load_graph("../graph.gpickle")

adj_mat = nx.adjacency_matrix(graph.graph)
nodes = graph.graph.nodes

max_out = -1
counter = 0
for n in nodes:
    curr_out = graph.graph.in_degree[n]
    if curr_out > 1:
        counter += 1

print(max_out, counter)