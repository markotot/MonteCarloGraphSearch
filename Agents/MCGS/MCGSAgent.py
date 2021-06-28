from Agents.AbstractAgent import AbstractAgent
from Agents.MCGS.Graph import Graph
from Utils.StateDatabase import StateDatabase
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

        if parent is None or MCGSAgent.config['inherit_novelty'] is False:
            self.novelty_value = novelty_value
        else:
            self.novelty_value = self.parent.novelty_value * MCGSAgent.config['inherit_novelty_factor'] + novelty_value

        self.chosen = False
        self.unreachable = False

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
            action_trajectory.insert(0, current_node.action)
            current_node = current_node.parent

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

    config = None

    def __init__(self, env, seed, config, verbose):
        super().__init__(env=env, seed=seed, verbose=verbose)
        MCGSAgent.config = config

        self.graph = Graph(seed, config)
        self.state_database = StateDatabase(config, self)

        self.node_counter = 0
        self.edge_counter = 0

        self.episodes = config['episodes']
        self.num_rollouts = config['num_rollouts']
        self.rollout_depth = config['rollout_depth']
        self.noisy_frontier_selection = config['noisy_frontier_selection']
        self.use_novelty = config['use_novelty']
        self.novelty_factor = config['novelty_factor']
        self.add_nodes_during_rollout = config['add_nodes_during_rollout']
        self.use_back_propagation = config['use_back_propagation']
        self.use_disabled_actions = config['use_disabled_actions']
        self.only_add_novel_states = config['only_add_novel_states']

        self.root_node = Node(ID=self.env.get_observation(), parent=None, is_leaf=True, done=False,
                              action=None, reward=0, visits=0, novelty_value=0)
        self.root_node.chosen = True
        self.node_counter += 1

        self.add_node(self.root_node)
        self.forward_model_calls = 0

        Logger.log_data(self.info(), time=False)
        Logger.log_data(f"Start: {str(self.agent_position(self.root_node))}")

    def plan(self, draw_graph=True) -> int:

        self.graph.set_root_node(self.root_node)
        self.graph.reroute_all()

        if self.verbose:
            iterations = trange(self.episodes, leave=True)
            iterations.set_description(
                f"Total: {str(len(self.graph.graph.nodes))} Frontier: {str(len(self.graph.frontier))}")
        else:
            iterations = range(self.episodes)

        for _ in iterations:
            selection_env = deepcopy(self.env)
            node = self.selection(selection_env)

            if node is None:
                children = self.graph.get_children(self.root_node)
            else:
                children = self.expansion(node, selection_env)

            for c in children:
                reward, novelties = self.simulation(c, selection_env)
                if self.add_nodes_during_rollout:
                    self.add_novelties_to_graph(novelties)
                if self.use_back_propagation:
                    self.back_propagation(c, reward)

        if draw_graph:
            self.graph.draw_graph()

        action = self.get_optimal_action(self.root_node)

        Logger.log_data(f"Action: {self.env.agent_action_mapper(action):<16}"
                        f"Current position: {str(self.agent_position(self.root_node)):<12}")

        return action

    def selection(self, env):

        if self.root_node.is_leaf:
            return self.root_node

        node = self.graph.select_frontier_node(noisy=self.noisy_frontier_selection, novelty_factor=self.novelty_factor * int(self.use_novelty))
        if node is None:
            return None

        for action in node.trajectory_from_root():
            env.step(action)
            self.forward_model_calls += 1

        assert node.id == env.get_observation()

        return node

    def expansion(self, node, env):

        node.is_leaf = False
        self.graph.remove_from_frontier(node)
        new_nodes = []

        for action in range(self.env.action_space.n):

            expansion_env = deepcopy(env)
            state, reward, done, _ = expansion_env.step(action)
            self.forward_model_calls += 1
            current_observation = expansion_env.get_observation()

            child = self.add_new_observation(current_observation, node, action, reward, done)
            if child is not None:
                new_nodes.append(child)

        return new_nodes

    def simulation(self, node, env):
        rewards = []
        total_steps = 0
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            paths = [None] * self.num_rollouts
            for i in range(self.num_rollouts):
                disabled_actions = []
                if self.use_disabled_actions:
                    if i < self.env.action_space.n:
                        disabled_actions.append(i)

                possible_actions = [x for x in range(self.env.action_space.n) if x not in disabled_actions]
                action_list = self.random.choice(possible_actions, self.rollout_depth)
                futures.append(executor.submit(self.rollout, node, env, action_list, i))

            for f in concurrent.futures.as_completed(futures):
                average_reward, path, i = f.result()
                paths[i] = path
                rewards.append(average_reward)
                total_steps += len(path)

        self.forward_model_calls += total_steps
        return np.mean(rewards), paths

    def rollout(self, node, env, action_list, i):
        cum_reward = 0
        path = []
        rollout_env = deepcopy(env)
        env.step(node.action)

        previous_observation = rollout_env.get_observation()
        for action in action_list:

            state, r, done, _ = rollout_env.step(action)
            observation = rollout_env.get_observation()
            cum_reward += r

            path.append((previous_observation, observation, action, r, done))
            previous_observation = observation
            if done:
                break

        return cum_reward, path, i

    def back_propagation(self, node, reward):
        y = 1
        while node is not None:
            node.visits += 1
            node.total_value += reward * y
            node = node.parent
            y *= y

    def add_novelties_to_graph(self, paths):

        for path in paths:
            for idx, step in enumerate(path):

                observation = step[1]
                novelty = self.state_database.calculate_novelty(observation)
                if self.graph.has_node(observation) is False and (novelty > 0 or self.only_add_novel_states is False):

                    for i in range(idx + 1):
                        step_i = path[i]
                        previous_observation = step_i[0]
                        current_observation = step_i[1]
                        action = step_i[2]
                        reward = step_i[3]
                        done = step_i[4]
                        novelty = self.state_database.calculate_novelty(current_observation)
                        parent_node = self.graph.get_node_info(previous_observation)

                        self.add_new_observation(current_observation, parent_node, action, reward, done)

                    if novelty >= 1:
                        node = self.graph.get_node_info(observation)
                        Logger.log_novel_data(f"Novel: {self.agent_position(node)}")

    def add_new_observation(self, current_observation, parent_node, action, reward, done):

        new_node = None
        if current_observation != parent_node.id:  # don't add node if nothing has changed in the observation
            if self.graph.has_node(current_observation) is False:  # if the node is new, create it and add to the graph
                child = Node(ID=current_observation, parent=parent_node,
                             is_leaf=True, done=done, action=action, reward=reward, visits=0,
                             novelty_value=self.state_database.calculate_novelty(current_observation))
                self.add_node(child)
                new_node = child
            else:  # if the node exists, get it
                child = self.graph.get_node_info(current_observation)
                if child.unreachable is True and child != self.root_node:  # if child was unreachable make it reachable through this parent
                    child.parent = parent_node
                    child.action = action
                    child.unreachable = False

            edge = Edge(ID=self.edge_counter, node_from=parent_node, node_to=child,
                        action=action, reward=reward, done=done)
            self.add_edge(edge)

            return new_node

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

        best_node = self.graph.get_best_node(only_reachable=True)
        if best_node.done is True:
            self.state_database.goal_found()

        while best_node.parent != self.root_node:
            best_node = best_node.parent

        edge = self.graph.get_edge_info(node, best_node)  # pick the edge between children
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
        env_name = self.env.name
        episodes = "Episodes: " + str(self.episodes)
        rollouts = "Num rollouts: " + str(self.num_rollouts)
        depth = "Depth: " + str(self.rollout_depth)
        seed = "Seed: " + str(self.env.seed)
        return [env_name, seed, episodes, rollouts, depth]

    def add_node(self, node):

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

    def get_metrics(self):

        metrics = dict(
            total_nodes=len(self.graph.graph.nodes),
            frontier_nodes=len(self.graph.frontier),
            forward_model_calls=self.forward_model_calls,
            key_found_nodes=self.state_database.subgoals['key_subgoal'][0],
            key_found_FMC=self.state_database.subgoals['key_subgoal'][1],
            door_found_nodes=self.state_database.subgoals['door_subgoal'][0],
            door_found_FMC=self.state_database.subgoals['door_subgoal'][1],
            goal_found_nodes=self.state_database.subgoals['goal_found'][0],
            goal_found_FMC=self.state_database.subgoals['goal_found'][1],
             )

        return metrics
