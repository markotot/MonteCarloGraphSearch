from Agents.MCGS.Graph import Graph
import networkx as nx
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def calculate_statistics(values):
    return np.mean(values), np.std(values), np.min(values), np.max(values)


def print_statistics(nodes):

    x_pos = [x.id[0] for x in nodes]
    y_pos = [x.id[1] for x in nodes]
    rotation = [x.id[2] for x in nodes]
    has_key = [0 if x.id[3] is None else 1 for x in nodes]

    x_pos_mean, x_pos_std, x_pos_min, x_pos_max = calculate_statistics(x_pos)
    y_pos_mean, y_pos_std, y_pos_min, y_pos_max = calculate_statistics(y_pos)
    rotation_mean, rotation_std, rotation_min, rotation_max = calculate_statistics(rotation)
    has_key_mean, has_key_std, has_key_min, has_key_max = calculate_statistics(has_key)

    print(f"x_position\t mean:{round(x_pos_mean, 2):<6} \t std:{round(x_pos_std, 2):<6} "
          f" \t min:{round(x_pos_min, 2):<6} \t max:{round(x_pos_max, 2):<6}")

    print(f"y_position\t mean:{round(y_pos_mean, 2):<6} \t std:{round(y_pos_std, 2):<6} "
          f" \t min:{round(y_pos_min, 2):<6} \t max:{round(y_pos_max, 2):<6}")

    print(f"rotation\t mean:{round(rotation_mean, 2):<6} \t std:{round(rotation_std, 2):<6}"
          f" \t min:{round(rotation_min, 2):<6} \t max:{round(rotation_max, 2):<6}")

    print(f"has_key \t mean:{round(has_key_mean, 2):<6} \t std:{round(has_key_std, 2):<6}"
          f" \t min:{round(has_key_min, 2):<6} \t max:{round(has_key_max, 2):<6}")


def create_exploration_heatmap(nodes, width, height):
    exploration_matrix = np.zeros(shape=[height, width])
    for node in nodes:
        x_value = node.id[0]
        y_value = node.id[1]
        exploration_matrix[y_value][x_value] += 1
    ax = sns.heatmap(exploration_matrix)
    plt.show()
    plt.close()

def create_novelty_heatmap(nodes, width, height, use_value=False):

    exploration_matrix = np.zeros(shape=[height, width])
    novelty_matrix = np.zeros(shape=[height, width])
    for node in nodes:
        x_value = node.id[0]
        y_value = node.id[1]
        exploration_matrix[y_value][x_value] += 1
        #novelty_matrix[y_value][x_value] += node.novelty_value
        if use_value:
            novelty_matrix[y_value][x_value] += node.total_value

    for y in range(height):
        for x in range(width):
            if exploration_matrix[y][x] != 0:
                novelty_matrix[y][x] /= exploration_matrix[y][x]

    ax = sns.heatmap(novelty_matrix)
    plt.show()
    plt.close()

config = {
    'amplitude_factor': 0,
    'noisy_min_value': 0,
}
path = "../graph.gpickle"
graph = Graph(seed=0, config=config)
graph.load_graph(path)

nodes = nx.get_node_attributes(graph.graph, 'info').values()

print_statistics(nodes)
#create_exploration_heatmap(nodes, width=16, height=16)
create_novelty_heatmap(nodes, width=16, height=16)
create_novelty_heatmap(nodes, width=16, height=16, use_value=True)




