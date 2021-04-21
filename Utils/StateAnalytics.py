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
x_pos_min = np.min(x_pos)
x_pos_max = np.max(x_pos)

y_pos_mean = np.mean(y_pos)
y_pos_std = np.std(y_pos)
y_pos_min = np.min(y_pos)
y_pos_max = np.max(y_pos)

rotation_mean = np.mean(rotation)
rotation_std = np.std(rotation)
rotation_min = np.min(rotation)
rotation_max = np.max(rotation)

has_key_mean = np.mean(has_key)
has_key_std = np.std(has_key)
has_key_min = np.min(has_key)
has_key_max = np.max(has_key)

print(f"x_position\t mean:{round(x_pos_mean, 2):<6} \t std:{round(x_pos_std, 2):<6} "
      f" \t min:{round(x_pos_min, 2):<6} \t max:{round(x_pos_max, 2):<6}")

print(f"y_position\t mean:{round(y_pos_mean, 2):<6} \t std:{round(y_pos_std, 2):<6} "
      f" \t min:{round(y_pos_min, 2):<6} \t max:{round(y_pos_max, 2):<6}")

print(f"rotation\t mean:{round(rotation_mean, 2):<6} \t std:{round(rotation_std, 2):<6}"
      f" \t min:{round(rotation_min, 2):<6} \t max:{round(rotation_max, 2):<6}")

print(f"has_key \t mean:{round(has_key_mean, 2):<6} \t std:{round(has_key_std, 2):<6}"
      f" \t min:{round(has_key_min, 2):<6} \t max:{round(has_key_max, 2):<6}")


