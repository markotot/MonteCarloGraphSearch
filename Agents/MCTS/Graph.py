import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from Utils.Logger import Logger
from networkx.drawing.nx_agraph import graphviz_layout

#Colors:
# orange    -   standard
# red       -   frontier
# blue      -   root
# cyan      -   previous roots
# green     -   done
# black     -   unreachable


class Graph:

    def __init__(self):
        self.graph = nx.DiGraph()
        self.root_node = None
        self.new_nodes = []

    def add_node(self, node):
        self.graph.add_node(node.id, info=node)
        self.new_nodes.append(node)

    def add_edge(self, edge):
        self.graph.add_edge(edge.node_from.id, edge.node_to.id, info=edge)

    def save_graph(self, path):
        nx.readwrite.write_gpickle(self.graph, path)

    def load_graph(self, path):
        self.graph = nx.readwrite.read_gpickle(path)

    def get_node_info(self, id):
        return self.graph.nodes[id]["info"]

    def get_nodes_with_degree(self, degree):
        node_list = []
        for node, out_degree in self.graph.out_degree():
            if self.get_node_from_observation(node).is_terminal:  # Poor optimization here for large graph
                continue
            if out_degree == degree or (out_degree == degree + 1 and self.graph.has_edge(node, node)):
                node_list.append(self.graph.nodes[node]["info"])
        return node_list

    def has_node(self, ID):
        return self.graph.has_node(ID)

    def has_edge(self, edge):
        parent = edge.node_from
        child = edge.node_to
        return self.graph.has_edge(parent.id, child.id)

    def has_edge_by_nodes(self, node_from, node_to):
        return self.graph.has_edge(node_from, node_to)

    def get_children(self, node):
        node_list = []
        for n in self.graph.successors(node.id):
            child_node = self.graph.nodes[n]["info"]
            node_list.append(child_node)

        return node_list

    def get_children_with_id(self, id):
        node_list = []
        for n in self.graph.successors(id):
            node_list.append(self.graph.nodes[n]["info"])
        return node_list

    def get_edge_info(self, parent, child):
        return self.graph.get_edge_data(parent.id, child.id)["info"]

    def draw_graph(self):

        nodes_info = nx.get_node_attributes(self.graph, 'info')
        node_color_map = []
        node_size_map = []
        value_map = {}

        for node in nodes_info.values():

            if node.is_leaf:
                value_map[node.id] = str(round(node.uct_value(), 2))
                node_size_map.append(30)
            else:
                value_map[node.id] = str(round(node.uct_value(), 2))
                node_size_map.append(30)

            if node == self.root_node:
                node_color_map.append('blue')
            elif node.chosen:
                node_color_map.append('lightblue')
            elif node.done:
                node_color_map.append('green')
            elif node.is_leaf:
                node_color_map.append('red')
            elif node.observation[3] == 'key':
                node_color_map.append('black')
            elif node in self.new_nodes:
                node_color_map.append('pink')
            else:
                node_color_map.append('orange')

        edges_info = nx.get_edge_attributes(self.graph, 'info')
        edge_color_map = []
        edge_width_map = []
        for edge in edges_info.values():
            if edge.node_from == self.root_node:
                edge_width_map.append(1)
                edge_color_map.append('blue')
            elif edge.node_to.chosen and edge.node_from.chosen:
                edge_width_map.append(1)
                edge_color_map.append('lightblue')
            else:
                edge_width_map.append(0.2)
                edge_color_map.append('grey')

        self.new_nodes.clear()

        general_options = {
            "with_labels": False,
            "font_size": 15,
        }

        node_options = {
            "node_color": node_color_map,
            "node_size": node_size_map,
        }

        edge_options = {
            "edge_color": edge_color_map,
            "width": edge_width_map,
            "arrowsize": 10,
        }

        pos = graphviz_layout(self.graph)
        options = {}

        options.update(general_options)
        options.update(node_options)
        options.update(edge_options)

        dpi = 96
        plt.figure(1, figsize=(1024/dpi, 768/dpi))

        nx.draw(self.graph, pos, **options)
        nx.draw_networkx_labels(self.graph, pos, value_map, font_size=8)
        plt.show()

    def save_graph(self, path):
        nx.readwrite.write_gpickle(self.graph, path + ".gpickle")
