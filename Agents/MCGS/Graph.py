import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

#Colors:
# orange    -   standard
# red       -   frontier
# blue      -   root
# cyan      -   previous roots
# green     -   done
# black     -   unreachable


class Graph:

    def __init__(self, seed, config):

        self.graph = nx.DiGraph()
        self.config = config
        self.frontier = []

        self.amplitude_factor = config['amplitude_factor']
        self.noisy_min_value = config['noisy_min_value']
        self.use_novelty_for_best_step = config['use_novelty_for_best_step']
        self.root_node = None
        self.new_nodes = []
        self.random = np.random.RandomState(seed)

    def add_node(self, node):
        self.graph.add_node(node.id, info=node)

    def add_edge(self, edge):
        self.graph.add_edge(edge.node_from.id, edge.node_to.id, info=edge)

    def add_to_frontier(self, node):
        self.frontier.append(node)
        self.new_nodes.append(node)

    def remove_from_frontier(self, node):
        self.frontier.remove(node)

    def in_frontier(self, node):
        return node in self.frontier

    def save_graph(self, path):
        nx.readwrite.write_gpickle(self.graph, path)

    def load_graph(self, path):
        self.graph = nx.readwrite.read_gpickle(path)

    def select_frontier_node(self, noisy, novelty_factor):

        selectable_nodes = [x for x in self.frontier if x.unreachable is False]
        if len(selectable_nodes) == 0:
            return None
        else:

            if noisy:
                amplitude = self.get_best_node(only_reachable=True).uct_value() * self.amplitude_factor
                noise = self.random.normal(0, max(amplitude, self.noisy_min_value), len(selectable_nodes))
            else:
                noise = 0

            best_node = selectable_nodes[0]
            best_node_value = best_node.uct_value() + noise[0] + novelty_factor * best_node.novelty_value
            for i, n in enumerate(selectable_nodes):
                if n.uct_value() + noise[i] + novelty_factor * n.novelty_value > best_node_value:
                    best_node = n
                    best_node_value = n.uct_value() + noise[i] + novelty_factor * n.novelty_value

            assert self.has_path(self.root_node, best_node)
            return best_node

    def set_root_node(self, root_node):
        self.root_node = root_node

    def reroute_paths(self, root_node):

        for node_id, node in self.graph.nodes.data('info'):
            if root_node.id != node_id:
                if self.has_path(self.root_node, node):
                    self.reroute_path(self.root_node, node)
                    node.unreachable = False
                else:
                    node.unreachable = True

    def reroute_path(self, node_from, node_to):
        nodes, actions = self.get_path(node_from, node_to)
        node_path = [self.get_node_info(x) for x in nodes]
        node_to.reroute(node_path, actions)

    def get_path(self, node_from, node_to):
        observations = nx.dijkstra_path(self.graph, node_from.id, node_to.id)
        actions = []
        for i in range(len(observations) - 1):
            actions.append(self.get_edge_info(self.get_node_info(observations[i]), self.get_node_info(observations[i + 1])).action)

        return observations, actions

    def get_path_length(self, node_from, node_to):
        nodes = nx.dijkstra_path(self.graph, node_from.id, node_to.id)
        return len(nodes)

    def has_path(self, node_from, node_to):
        return nx.has_path(self.graph, node_from.id, node_to.id)

    def get_node_info(self, id):
        return self.graph.nodes[id]["info"]

    def get_all_nodes_info(self):
        return list(nx.get_node_attributes(self.graph, 'info').values())

    def get_nodes_with_degree(self, degree):
        node_list = []
        for node, out_degree in self.graph.out_degree():
            if self.get_node_from_observation(node).is_terminal:  # Poor optimization here for large graph
                continue
            if out_degree == degree or (out_degree == degree + 1 and self.graph.has_edge(node, node)):
                node_list.append(self.graph.nodes[node]["info"])
        return node_list

    def get_best_node(self, only_reachable=False):

        nodes = self.get_all_nodes_info()
        nodes.remove(self.root_node)

        if only_reachable:
            selectable_nodes = [x for x in nodes if x.unreachable is False]
        else:
            selectable_nodes = nodes

        if len(selectable_nodes) > 0:
            best_node = selectable_nodes[0]
            best_node_value = best_node.value() + self.get_edge_info(best_node.parent, best_node).reward
            if self.use_novelty_for_best_step:
                best_node_value += best_node.novelty_value
        else:
            best_node = None
            best_node_value = None

        for n in selectable_nodes:
            selected_node_value = n.value() + self.get_edge_info(n.parent, n).reward
            if self.use_novelty_for_best_step:
                selected_node_value += n.novelty_value
            if best_node_value < selected_node_value:
                best_node = n
                best_node_value = selected_node_value

        return best_node

    def get_closest_done_node(self, only_reachable=False):

        selectable_nodes = [x for x in self.get_all_nodes_info() if x.done and x.unreachable is False]
        if only_reachable:
            selectable_nodes = [x for x in selectable_nodes if x.unreachable is False]

        if len(selectable_nodes) > 0:
            best_node = selectable_nodes[0]
            best_node_length = self.get_path_length(self.root_node, best_node)
        else:
            best_node = None
            best_node_length = None

        for n in selectable_nodes:
            selected_node_length = self.get_path_length(self.root_node, n)
            if selected_node_length < best_node_length:
                best_node = n
                best_node_length = selected_node_length

        return best_node

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

    def get_child_with_action(self, id, action):
        for edge in self.graph.out_edges(id, data=True):
            if edge[2]["info"].action == action:
                return edge[1]  # return child node
        return None


    def get_edge_info(self, parent, child):
        return self.graph.get_edge_data(parent.id, child.id)["info"]

    def draw_graph(self):

        nodes_info = nx.get_node_attributes(self.graph, 'info')
        node_color_map = []
        node_size_map = []
        value_map = {}

        for node in nodes_info.values():

            if (node.novelty_value == 0 and node.value() == 0) or node not in self.frontier:
                value_map[node.id] = ""
            else:
                #value_map[node.id] = str(round(node.novelty_value + node.value(), 2))
                value_map[node.id] =  ""
            node_size_map.append(30)

            if node == self.root_node:
                node_color_map.append('blue')
            elif node.chosen:
                node_color_map.append('lightblue')
            elif node.unreachable:
                node_color_map.append('grey')
            elif node in self.new_nodes:
                node_color_map.append('pink')
            elif node.done:
                node_color_map.append('green')
            elif node in self.frontier:
                node_color_map.append('red')
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

        pos = graphviz_layout(self.graph, prog='neato')

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

    def reroute_all(self):
        i = 0
        all_nodes = self.get_all_nodes_info()
        for n in all_nodes:
            n.unreachable = True

        # BFS implementation
        visited = []
        queue = []
        root_node_id = self.root_node.id

        visited.append(root_node_id)
        queue.append(root_node_id)

        while queue:
            node_id = queue.pop(0)
            node = self.get_node_info(node_id)
            for child in self.graph.successors(node_id):
                # Set all of the new routes
                if child not in visited:
                    child_node = self.get_node_info(child)
                    child_node.unreachable = False
                    child_node.parent = node
                    child_node.action = self.get_edge_info(node, child_node).action
                    i += 1
                    visited.append(child)
                    queue.append(child)

    def reroute_all_optimized(self):
        i = 0
        all_nodes = self.get_all_nodes_info()
        for n in all_nodes:
            n.unreachable = True

        # BFS implementation
        visited = []
        queue = []
        root_node_id = self.root_node.id

        visited.append(root_node_id)
        queue.append(root_node_id)

        while queue:
            node_id = queue.pop(0)
            node = self.get_node_info(node_id)
            for child in self.graph.successors(node_id):
                # Set all of the new routes
                if child not in visited:
                    child_node = self.get_node_info(child)
                    child_node.unreachable = False
                    if child_node.parent == node:
                        pass
                    else:
                        child_node.parent = node
                        child_node.action = self.get_edge_info(node, child_node).action
                        queue.append(child)
                    visited.append(child)
                    i += 1
        #print(i)


