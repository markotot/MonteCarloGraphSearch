import networkx as nx
import matplotlib.pyplot as plt

from networkx.drawing.nx_agraph import graphviz_layout


def agent_rotation_mapper(agent_dir):
    return {0: "right", 1: "down", 2: "left", 3: "up"}[agent_dir]

def draw_graph(root_node, graph):
    nodes_info = nx.get_node_attributes(graph, 'info')
    node_color_map = []
    node_size_map = []
    value_map = {}

    for node in nodes_info.values():

        agent_pos_x = node.state.env.agent_pos[0]
        agent_pos_y = node.state.env.agent_pos[1]
        agent_dir = agent_rotation_mapper(node.state.env.agent_dir)

        value_map[node] = f"({agent_pos_x}, {agent_pos_y}, {agent_dir})"
        # if (node.novelty_value == 0 and node.value() == 0) or node not in frontier:
        #     value_map[node.id] = ""
        # else:
        #     value_map[node.id] = str(round(node.novelty_value + node.value(), 2))
        node_size_map.append(30)

        if node == root_node:
            node_color_map.append('blue')
        elif node.terminal:
            node_color_map.append('red')
        else:
            node_color_map.append('orange')

    # edges_info = nx.get_edge_attributes(graph, 'info')
    # edge_color_map = []
    # edge_width_map = []
    # for edge in edges_info.values():
    #     if edge.node_from == root_node:
    #         edge_width_map.append(1)
    #         edge_color_map.append('blue')
    #     elif edge.node_to.chosen and edge.node_from.chosen:
    #         edge_width_map.append(1)
    #         edge_color_map.append('lightblue')
    #     else:
    #         edge_width_map.append(0.2)
    #         edge_color_map.append('grey')

    general_options = {
        "with_labels": False,
        "font_size": 15,
    }

    node_options = {
        "node_color": node_color_map,
        "node_size": node_size_map,
    }

    # edge_options = {
    #     "edge_color": edge_color_map,
    #     "width": edge_width_map,
    #     "arrowsize": 10,
    # }

    pos = graphviz_layout(graph, prog='neato')

    options = {}
    options.update(general_options)
    options.update(node_options)
    # options.update(edge_options)

    dpi = 96
    plt.figure(1, figsize=(1024 / dpi, 768 / dpi))

    nx.draw(graph, pos, **options)
    nx.draw_networkx_labels(graph, pos,
                            value_map,
                            font_size=12)
    plt.show()