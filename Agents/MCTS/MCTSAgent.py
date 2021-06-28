from Agents.AbstractAgent import AbstractAgent
from Agents.MCTS.Graph import Graph

from Utils.Logger import Logger
from Utils.StateDatabase import StateDatabase
from math import sqrt, log
from copy import deepcopy
from tqdm import trange

import numpy as np
import concurrent.futures


class Node:

    def __init__(self, ID, parent, is_leaf, done, value, visits, observation):

        self.id = ID
        self.parent = parent
        self.done = done
        self.value = value
        self.visits = visits
        self.is_leaf = is_leaf
        self.observation = observation

        self.not_reachable = False
        self.chosen = False

    def uct_value(self):

        c = 0.001
        mean = self.value / self.visits
        ucb = c * sqrt(log(self.parent.visits if self.parent is not None else 1 + 1) / self.visits)
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

    config = None

    def __init__(self, env, seed, config, verbose):
        super().__init__(env=env, seed=seed, verbose=verbose)
        MCTSAgent.config = config

        self.graph = Graph()
        self.state_database = StateDatabase(config, self)

        self.node_counter = 0
        self.edge_counter = 0

        self.episodes = config['episodes']
        self.num_rollouts = config['num_rollouts']
        self.rollout_depth = config['rollout_depth']

        self.root_node = Node(ID=self.node_counter, parent=None, is_leaf=True, done=False, value=0, visits=0, observation=self.env.get_observation())
        self.root_node.chosen = True
        self.node_counter += 1
        self.add_node(self.root_node)

        self.forward_model_calls = 0

    def plan(self, draw_graph=True):

        iterations = trange(self.episodes, leave=True)
        iterations.set_description(
            f"Total: {str(len(self.graph.graph.nodes))}")

        for i in iterations:
            selection_env = deepcopy(self.env)

            node = self.selection(self.root_node, selection_env)

            children = self.expansion(node, selection_env)

            for c in children:
                value = self.simulation(c, selection_env)
                self.back_propagation(c, value)

        if draw_graph:
            self.graph.draw_graph()
        print(f"Current expected value: {self.root_node.value / self.root_node.visits}")

        optimal_action = self.get_optimal_action(self.root_node)
        return optimal_action

    def learn(self):
        pass

    def selection(self, node, env):

        while not node.is_leaf:
            node, action = self.select_child(node, criteria="uct")
            env.step(action)
            self.forward_model_calls += 1

        return node

    def expansion(self, node, env):
        children = []
        node.is_leaf = False

        for a in range(self.env.action_space.n):
            expansion_env = deepcopy(env)
            state, reward, done, _ = expansion_env.step(a)
            self.forward_model_calls += 1
            observation = expansion_env.get_observation()
            child = Node(ID=self.node_counter, parent=node, is_leaf=True, done=done, value=0, visits=0, observation=observation)
            edge = Edge(ID=self.edge_counter, node_from=node, node_to=child, action=a, reward=reward, done=done)

            self.node_counter += 1
            self.edge_counter += 1

            self.add_node(child)
            children.append(child)

            self.graph.add_edge(edge)

        return children

    def simulation(self, node, env):
        rewards = []
        total_steps = 0
        action = self.graph.get_edge_info(node.parent, node).action
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []

            for i in range(self.num_rollouts):
                possible_actions = [x for x in range(self.env.action_space.n)]
                action_list = self.random.choice(possible_actions, self.rollout_depth)
                futures.append(executor.submit(self.rollout, action, env, action_list, i))

            for f in concurrent.futures.as_completed(futures):
                average_reward, step_counter, i = f.result()
                rewards.append(average_reward)
                total_steps += step_counter

        self.forward_model_calls += total_steps
        return np.mean(rewards)

    def rollout(self, action, env, action_list, i):
        cum_reward = 0
        step_counter = 0
        rollout_env = deepcopy(env)
        env.step(action)

        for a in action_list:
            state, r, done, _ = rollout_env.step(a)
            cum_reward += r
            step_counter += 1
            if done:
                break

        return cum_reward, step_counter, i

    def back_propagation(self, node, value):
        y = 1
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
        self.root_node = new_root_node

        return action

    def add_node(self, node):
        Logger.log_graph_data(node.observation)
        self.graph.add_node(node)
        self.state_database.update_posterior(node.observation)

    def select_child(self, node, criteria="uct"):

        children = self.graph.get_children(node)  # get children
        edges = []
        for c in children:
            edges.append(self.graph.get_edge_info(node, c))

        if criteria == "uct":
            children_criteria = [x.uct_value() for x in children]  # find their uct values
        elif criteria == "value":
            children_criteria = [(x.value / (x.visits + 1) + self.graph.get_edge_info(node, x).reward) for x in children]  # find their Q values

        child = children[children_criteria.index(max(children_criteria))]  # pick the child with max uct
        edge = self.graph.get_edge_info(node, child)  # pick the edge between children

        return child, edge.action

    def agent_position(self, node):
        agent_pos_x = node.id[0]
        agent_pos_y = node.id[1]
        agent_dir = self.env.agent_rotation_mapper(node.id[2])
        agent_has_key = node.id[3]
        agent_door_open = node.id[4]
        agent_door_locked = node.id[5]
        return tuple([agent_pos_x, agent_pos_y, agent_dir, agent_has_key, agent_door_open, agent_door_locked])
