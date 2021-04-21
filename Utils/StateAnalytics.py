from Agents.MCGS.Graph import Graph
import networkx as nx
import numpy as np

path = "../graph.gpickle"

graph = Graph()
graph.load_graph(path)

nodes = list(nx.get_node_attributes(graph.graph, 'info').keys())



x_pos = [x[0] for x in nodes]
y_pos = [x[1] for x in nodes]
rotation = [x[2] for x in nodes]
has_key = [0 if x[3] is None else 1 for x in nodes]

x_pos_mean = np.mean(x_pos)
x_pos_std = np.std(x_pos)

y_pos_mean = np.mean(y_pos)
y_pos_std = np.std(y_pos)

rotation_mean = np.mean(rotation)
rotation_std = np.std(rotation)

has_key_mean = np.mean(has_key)
has_key_std = np.std(has_key)

print(f"x position\t mean:{round(x_pos_mean, 2)} \t std:{round(x_pos_std, 2)}")
print(f"y position\t mean:{round(y_pos_mean, 2)} \t std:{round(y_pos_std, 2)}")
print(f"rotation  \t mean:{round(rotation_mean, 2)} \t std:{round(rotation_std, 2)}")
print(f"has_key  \t mean:{round(has_key_mean, 2)} \t std:{round(has_key_std, 2)}")


