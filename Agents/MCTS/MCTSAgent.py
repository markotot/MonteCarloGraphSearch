from Algorithms.AlgorithmWrapper import AbstractAgent
from Algorithms.Graph import Graph
from Utils.Logger import Logger
from math import sqrt, log
import numpy as np
from copy import deepcopy
from tqdm import trange


from operator import add


class Node:

    def __init__(self, ID, parent, is_leaf, done, value, visits):

        self.id = ID
        self.parent = parent
        self.done = done
        self.value = value
        self.visits = visits
        self.is_leaf = is_leaf

        self.not_reachable = False
        self.chosen = False

    def uct_value(self):

        c = 0.0
        if self.is_leaf:
            return 1
        else:
            mean = self.value / self.visits
            ucb = c * sqrt(log(self.parent.visits + 1) / self.visits)
            return mean + ucb

    def __hash__(self):
        return hash(self.id)


class Edge:

    def __init__(self, ID, node_from, node_to, action, reward, done):

        self.id = ID
        self.node_from = node_from
        self.node_to = node_to
        self.action = action
        self.reward = reward
        self.done = done

    def __hash__(self):
        return hash(self.id)


class MCTSAgent(AbstractAgent):

    def __init__(self, env, episodes, num_rollouts, rollout_depth):
        super().__init__(env)

        self.graph = Graph()

        self.node_counter = 0
        self.edge_counter = 0

        self.episodes = episodes
        self.num_rollouts = num_rollouts
        self.rollout_depth = rollout_depth

        self.root_node = Node(ID=self.node_counter, parent=None, is_leaf=True, done=False, value=0, visits=0)
        self.root_node.chosen = True
        self.node_counter += 1
        self.graph.add_node(self.root_node)

    def plan(self, draw_graph=True):

        iterations = trange(self.episodes, leave=True)

        for i in iterations:
            selection_env = deepcopy(self.env)
            node = self.selection(self.root_node, selection_env)
            self.expansion(node, selection_env)
            value = self.simulation(selection_env)
            self.back_propagation(node, value)
        if draw_graph:
            self.graph.draw_graph()
        print(f"Current expected value: {self.root_node.value / self.root_node.visits}")
        return self.get_optimal_action(self.root_node)

    def learn(self):
        pass

    def selection(self, node, env):

        while not node.is_leaf:
            node, action = self.select_child(node, criteria="uct")
            env.step(action)
        return node

    def expansion(self, node, env):

        node.is_leaf = False

        for a in range(self.env.action_space.n):
            expansion_env = deepcopy(env)
            state, reward, done, _ = expansion_env.step(a)

            child = Node(ID=self.node_counter, parent=node, is_leaf=True, done=done, value=0, visits=0)
            edge = Edge(ID=self.edge_counter, node_from=node, node_to=child, action=a, reward=reward, done=done)

            self.node_counter += 1
            self.edge_counter += 1

            self.graph.add_node(child)
            self.graph.add_edge(edge)

    def simulation(self, env):

        rewards = []
        for _ in range(self.num_rollouts):
            simulation_env = deepcopy(env)  # TODO: either find a way to copy, or.... swtich to MiniGrid
            cum_reward = 0
            for t in range(self.rollout_depth):
                _, r, done, _ = simulation_env.random_step()
                if done:
                    break
            cum_reward += r
            rewards.append(cum_reward)
        return np.mean(rewards)

    def back_propagation(self, node, value):
        y = 0.95
        # while node is not None:
        while True:
            node.visits += 1
            node.value += value * y
            if node.chosen:
                break
            node = node.parent
            y *= y

    def get_optimal_action(self, node):
        new_root_node, action = self.select_child(node, criteria="value")
        new_root_node.chosen = True

        return action

    def select_child(self, node, criteria="uct"):

        children = self.graph.get_children(node)  # get children
        edges = []
        for c in children:
            edges.append(self.graph.get_edge_info(node, c))


        if criteria == "uct":
            children_criteria = [x.uct_value() for x in children]  # find their uct values
        elif criteria == "value":
            children_criteria = [(x.total_value / (x.visits + 1) + self.graph.get_edge_info(node, x).total_value) for x in children]  # find their Q values


        child = children[children_criteria.index(max(children_criteria))]  # pick the child with max uct
        edge = self.graph.get_edge_info(node, child)  # pick the edge between children

        return child, edge.action
