from Agents.AbstractAgent import AbstractAgent
from Agents.MCGS.Graph import Graph
from Agents.MCGS.StateDatabase import StateDatabase
from Utils.Logger import Logger

import numpy as np
import concurrent.futures

from copy import deepcopy
from tqdm import trange


class Node:

    def __init__(self, ID, parent, is_leaf, done, action, reward, visits, novelty_value):

        self.id = ID
        self.parent = parent

        if self.parent is not None:
            Logger.log_reroute_data(f" Ctor\n\tNode: {self.id[0:6]} Parent: {self.parent.id[0:6]}")
        self.done = done
        self.action = action
        self.total_value = reward
        self.visits = visits
        self.is_leaf = is_leaf

        self.novelty_value = novelty_value

        self.chosen = False
        self.not_reachable = False

    def uct_value(self):

        c = 0.0
        ucb = 0  # c * sqrt(log(self.parent.visits + 1) / self.visits)
        return self.value() + ucb

    def value(self):
        if self.visits == 0:
            return 0
        else:
            return self.total_value / self.visits

    def trajectory_from_root(self, debug=False):

        action_trajectory = []
        current_node = self

        while current_node.parent is not None:
            if debug:
                print(f"Node: {current_node.id} Parent: {current_node.parent.id} Action: {current_node.action}")
            action_trajectory.insert(0, current_node.action)
            current_node = current_node.parent

        if debug:
            print(action_trajectory)
        return action_trajectory

    def reroute(self, path, actions):
        parent_order = list(reversed(path))
        actions_order = list(reversed(actions))
        node = self
        Logger.log_reroute_data(f"Reroute: {self.id[0:5]}")
        for i in range(len(parent_order) - 1):
            node.parent = parent_order[i + 1]

            Logger.log_reroute_data(f"\tNode: {node.id[0:5]} Parent: {node.parent.id[0:5]}")
            node.action = actions_order[i]
            node = parent_order[i + 1]

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


class MCGSAgent(AbstractAgent):

    def __init__(self, env, episodes, num_rollouts, rollout_depth):
        super().__init__(env)

        self.graph = Graph()
        self.state_database = StateDatabase()

        self.node_counter = 0
        self.edge_counter = 0

        self.episodes = episodes
        self.num_rollouts = num_rollouts
        self.rollout_depth = rollout_depth

        self.root_node = Node(ID=self.env.get_observation(), parent=None, is_leaf=True, done=False,
                              action=None, reward=0, visits=0, novelty_value=0)
        self.root_node.chosen = True
        self.node_counter += 1

        self.add_node(self.root_node)

        Logger.log_data(self.info(), time=False)
        Logger.log_data(f"Start: {str(self.agent_position(self.root_node))}")

    def plan(self, draw_graph=True):

        # TODO: optimize dijkstra, do the checks only after a node has been selected for expansion
        self.graph.set_root(self.root_node)
        # uncomment to reroute everything every step (useful for drawing a pretty graph)
        # self.check_paths()

        # iterations = range(self.episodes)
        iterations = trange(self.episodes, leave=True)
        iterations.set_description(
            f"Total: {str(len(self.graph.graph.nodes))} Frontier: {str(len(self.graph.frontier))}")

        for _ in iterations:

            selection_env = deepcopy(self.env)
            node = self.selection(selection_env)

            if node is None:
                children = self.graph.get_children(self.root_node)
            else:
                children = self.expansion(node, selection_env)

            for c in children:
                reward, novelties = self.simulation(c, selection_env)
                self.add_novelties_to_graph(novelties)
                self.back_propagation(c, reward)

        if draw_graph:
            self.graph.draw_graph()

        action = self.get_optimal_action(self.root_node)
        self.graph.find_reachable()
        Logger.log_data(f"Action: {self.env.agent_action_mapper(action):<16}"
                        f"Current position: {str(self.agent_position(self.root_node)):<12}")

        return action

    def selection(self, env):

        if self.root_node.is_leaf:
            return self.root_node

        node = self.graph.select_frontier_node(noisy=True, novelty_factor=0.1)
        if node is None:
            return None

        for action in node.trajectory_from_root():
            env.step(action)

        assert node.id == env.get_observation()

        return node

    def expansion(self, node, env):

        node.is_leaf = False
        self.graph.remove_from_frontier(node)
        new_nodes = []

        for a in range(self.env.action_space.n):

            expansion_env = deepcopy(env)
            state, reward, done, _ = expansion_env.step(a)
            current_observation = expansion_env.get_observation()

            if current_observation != node.id:

                if self.graph.has_node(current_observation):
                    child = self.graph.get_node_info(current_observation)
                else:

                    child = Node(ID=current_observation, parent=node, is_leaf=True, done=done,
                                 action=a, reward=0, visits=0,
                                 novelty_value=self.state_database.calculate_novelty(current_observation))

                    self.add_node(child)
                    new_nodes.append(child)

                edge = Edge(ID=self.edge_counter, node_from=node, node_to=child, action=a, reward=reward, done=done)
                self.add_edge(edge)

        return new_nodes

    def simulation(self, node, env, disable_actions=True):
        rewards = []
        paths = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for i in range(self.num_rollouts):
                disabled_actions = []
                if disable_actions:
                    if i < self.env.action_space.n:
                        disabled_actions.append(i)
                futures.append(executor.submit(self.rollout, node, env, disabled_actions))

            for f in concurrent.futures.as_completed(futures):
                average_reward, path = f.result()
                rewards.append(average_reward)
                paths.append(path)

        return np.mean(rewards), paths

    def rollout(self, node, env, disabled_actions=[]):
        cum_reward = 0
        path = []
        rollout_env = deepcopy(env)
        env.step(node.action)

        previous_observation = rollout_env.get_observation()
        for n in range(self.rollout_depth):

            state, r, done, _ = rollout_env.random_step(disabled_actions)
            action = rollout_env.action
            observation = rollout_env.get_observation()
            cum_reward += r

            path.append((previous_observation, observation, action, r, done))
            previous_observation = observation
            if done:
                break

        return cum_reward, path

    def back_propagation(self, node, reward):
        y = 1
        while node is not None:
            node.visits += 1
            node.total_value += reward * y
            node = node.parent
            y *= y

    def add_novelties_to_graph(self, paths, only_novel=False):

        for path in paths:
            for idx, step in enumerate(path):

                observation = step[1]
                novelty = self.state_database.calculate_novelty(observation)
                if self.graph.has_node(observation) is False and (novelty > 0 or only_novel is False):

                    for i in range(idx + 1):
                        step_i = path[i]
                        previous_observation = step_i[0]
                        current_observation = step_i[1]
                        action = step_i[2]
                        reward = step_i[3]
                        done = step_i[4]
                        novelty = self.state_database.calculate_novelty(current_observation)

                        if self.graph.has_node(current_observation) is False:
                            parent = self.graph.get_node_info(previous_observation)
                            child = Node(ID=current_observation, parent=parent, is_leaf=True, done=done, action=action,
                                         reward=reward, visits=0,
                                         novelty_value=novelty)
                            self.add_node(child)
                            edge_i = Edge(ID=self.edge_counter, node_from=parent, node_to=child,
                                          action=action, reward=reward, done=done)
                            self.add_edge(edge_i, who="Roll")

                    if novelty > 0:
                        node = self.graph.get_node_info(observation)
                        Logger.log_novel_data(f"Novel: {self.agent_position(node)}")

    def get_optimal_action(self, node):

        new_root_node, action = self.select_best_step(node)
        new_root_node.chosen = True
        new_root_node.parent = None

        if self.graph.has_path(new_root_node, self.root_node):
            self.graph.reroute_path(new_root_node, self.root_node)
            self.root_node.action = self.graph.get_edge_info(self.root_node.parent, self.root_node).action

        self.root_node = new_root_node

        return action

    def select_best_step(self, node):

        children = self.graph.get_children(node)  # get children
        children_criteria = [(x.value() + self.graph.get_edge_info(node, x).reward) for x in
                             children]  # find their Q values

        # best_node = children[children_criteria.index(max(children_criteria))]  # pick the best child
        best_node = self.graph.get_best_node(only_reachable=True)
        print(f"Target: {self.agent_position(best_node)}: {round(best_node.value(), 5)}")
        Logger.log_data(f"Target: {self.agent_position(best_node)}: {round(best_node.value(), 5)}")

        while best_node.parent != self.root_node:
            best_node = best_node.parent

        edge = self.graph.get_edge_info(node, best_node)  # pick the edge between children

        Logger.log_data("Choices:")
        for i in range(len(children)):
            Logger.log_data(f"\t [{self.env.agent_action_mapper(self.graph.get_edge_info(node, children[i]).action)}]: "
                            f"{self.agent_position(children[i])}: {round(children_criteria[i], 5)}")

        return best_node, edge.action

    def check_paths(self):

        self.graph.reroute_paths(self.root_node)

    def agent_position(self, node):
        agent_pos_x = node.id[0]
        agent_pos_y = node.id[1]
        agent_dir = self.env.agent_rotation_mapper(node.id[2])
        agent_has_key = node.id[3]
        agent_door_open = node.id[4]
        agent_door_locked = node.id[5]
        return tuple([agent_pos_x, agent_pos_y, agent_dir, agent_has_key, agent_door_open, agent_door_locked])

    def info(self):

        env_name = self.env.env.unwrapped.spec.id
        episodes = "Episodes: " + str(self.episodes)
        rollouts = "Num rollouts: " + str(self.num_rollouts)
        depth = "Depth: " + str(self.rollout_depth)

        return [env_name, episodes, rollouts, depth]

    def add_node(self, node):

        # redundant, but useful for debugging
        assert not self.graph.has_node(node)
        if not self.graph.has_node(node):
            self.graph.add_node(node)

            if node.done is False:
                self.graph.add_to_frontier(node)
            self.node_counter += 1

            if node.parent is not None:
                Logger.log_graph_data(f"Child: {str(self.agent_position(node)):<12} "
                                      f" Parent: {str(self.agent_position(node.parent)):<12}"
                                      f" Action: {self.env.agent_action_mapper(node.action):<16}")

            self.state_database.update_posterior(node.id)

    def add_edge(self, edge, who="Expansion"):
        if not self.graph.has_edge(edge):

            self.graph.add_edge(edge)
            self.edge_counter += 1

            Logger.log_graph_data(f"{who} - New Edge: {str(self.agent_position(edge.node_from)):>12}"
                                  f" -> {str(self.agent_position(edge.node_to)):<12}"
                                  f" Action: {self.env.agent_action_mapper(edge.action):<16}")


