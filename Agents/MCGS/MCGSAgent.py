from Agents.AbstractAgent import AbstractAgent
from Agents.MCGS.Graph import Graph

from Environments.MyMinigridEnv import EnvType
from Novelty.DoorKeyNovelty import DoorKeyNovelty
from Novelty.EmptyNovelty import EmptyNovelty
from Utils.Logger import Logger

import numpy as np
import concurrent.futures
import time

from copy import deepcopy
from tqdm import trange


class Node:

    def __init__(self, ID, parent, is_leaf, done, action, reward, visits, novelty_value):

        self.id = ID
        self.parent = None
        self.set_parent(parent)

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

    def trajectory_from_root(self):

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

        for i in range(len(parent_order) - 1):

            if node.parent != parent_order[i + 1]:
                node.set_parent(parent_order[i + 1])
            node.action = actions_order[i]
            node = parent_order[i + 1]

    def set_parent(self, parent):
        self.parent = parent

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

        if env.env_type == EnvType.DoorKey:
            self.novelty_stats = DoorKeyNovelty(config, self)
        elif env.env_type == EnvType.Empty:
            self.novelty_stats = EmptyNovelty(config, self)
        else:
            raise ValueError

        self.node_counter = 0
        self.edge_counter = 0
        self.steps = 0

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

        self.start_node = self.root_node

    def plan(self, draw_graph=True) -> int:
        self.steps += 1


        # benchmark
        total_steps = 0
        times_selection = []
        times_expansion = []
        times_simulation = []
        times_backpropagation = []
        times_reroute = []

        times_thread = []
        times_randomness = []
        times_rollouts = []
        times_steps = []
        times_deepcopy = []


        start = time.perf_counter()    # benchmark
        self.set_root_node()
        self.graph.reroute_all()
        end = time.perf_counter()  # benchmark
        times_reroute.append(end - start)  # benchmark

        if self.verbose:
            iterations = trange(self.episodes, leave=True)
            iterations.set_description(
                f"Total: {str(len(self.graph.graph.nodes))} Frontier: {str(len(self.graph.frontier))}")
        else:
            iterations = range(self.episodes)

        for _ in iterations:

            start = time.perf_counter()  # benchmark
            selection_env = deepcopy(self.env)
            node = self.selection(selection_env)
            end = time.perf_counter()  # benchmark
            times_selection.append(end - start)  # benchmark

            if node is None:
                children, actions_to_children = [self.root_node], [6] # 6 is Done Action (Do nothing)
            else:
                start = time.perf_counter()  # benchmark
                children, actions_to_children = self.expansion(node, selection_env)
                end = time.perf_counter()  # benchmark
                times_expansion.append(end - start)  # benchmark

            for idx in range(len(children)):

                start = time.perf_counter()    # benchmark
                child_average_reward, novelties, simulation_steps, simulation_benchmarks = self.simulation(actions_to_children[idx], selection_env)

                times_deepcopy.append(simulation_benchmarks[0])
                times_rollouts.append(simulation_benchmarks[1])
                times_steps.append(simulation_benchmarks[2])
                times_randomness.append(simulation_benchmarks[3])
                times_thread.append(simulation_benchmarks[4])

                total_steps += simulation_steps
                end = time.perf_counter()  # benchmark
                times_simulation.append(end - start)  # benchmark

                if self.add_nodes_during_rollout:
                    rollout_nodes, rewards = self.add_novelties_to_graph(novelties)
                if self.use_back_propagation:
                    start = time.perf_counter()    # benchmark
                    if self.add_nodes_during_rollout:
                        for i, node in enumerate(rollout_nodes):
                            self.back_propagation(node, rewards[i])
                    self.back_propagation(children[idx], child_average_reward)
                    end = time.perf_counter()  # benchmark
                    times_backpropagation.append(end - start)  # benchmark

        if draw_graph:
            self.graph.draw_graph()

        best_node, action = self.select_best_step(self.root_node, closest=True)
        
        Logger.log_data(f"Current position: {str(self.agent_position(self.root_node)):<40}"
                        f"Action: {self.env.agent_action_mapper(action):<12}")

        Logger.log_reroute_data(f"Selection: {round(np.mean(times_selection), 4)},"
                                f" Expansion: {round(np.mean(times_expansion), 4)},"
                                f" Simulation ({total_steps}): {round(np.mean(times_simulation), 4)},"
                                f" Backprop: {round(np.mean(times_backpropagation), 4)}"
                                f" Reroute: {round(np.mean(times_reroute), 4)}")
        Logger.log_reroute_data(f"Threads: {round(np.mean(times_thread),4)}, "
                                f"Randomness: {round(np.mean(times_randomness), 4)}, "
                                f"Rollout: {round(np.mean(times_rollouts), 4)}, "
                                f"Step: {round(np.mean(times_steps), 4)}, "
                                f"Deepcopy:{round(np.mean(times_deepcopy), 4)}")
        return action

    def selection(self, env):

        if self.root_node.is_leaf:
            return self.root_node

        node = self.graph.select_frontier_node(noisy=self.noisy_frontier_selection, novelty_factor=self.novelty_factor * int(self.use_novelty))

        if node is None:
            return None

        obs_trajectory = [env.get_observation()]

        nodes_on_path, actions = self.graph.get_path(self.root_node, node)

        # TODO: For stochastic, continuously go to the node
        for idx, action in enumerate(node.trajectory_from_root()):

            previous_observation = env.get_observation()
            parent_node = self.graph.get_node_info(previous_observation)
            state, reward, done, _ = env.step(action)
            self.forward_model_calls += 1

            current_observation = env.get_observation()
            obs_trajectory.append(current_observation)
            if not self.graph.has_node(current_observation):
                self.add_new_observation(current_observation, parent_node, action, reward, done)
            elif not self.graph.has_edge_by_nodes(parent_node, self.graph.get_node_info(current_observation)):
                self.add_edge(parent_node, self.graph.get_node_info(current_observation), action, reward, done, who="Selection")

        selected_node = self.graph.get_node_info(env.get_observation())
        if self.graph.has_path(self.root_node, selected_node) is False:
            print("What2")
            assert False
        # assert node.id == env.get_observation()  # Must be true for deterministic (action_failure_prob == 0)

        return selected_node

    def expansion(self, node, env):

        if node.done:
            return [], []

        new_nodes = []
        actions_to_new_nodes = []
        # Nodes might not be leaves due to action_failure
        if node.is_leaf:
            node.is_leaf = False
            self.graph.remove_from_frontier(node)

        for action in range(self.env.action_space.n):

            expansion_env = deepcopy(env)
            state, reward, done, _ = expansion_env.step(action)
            self.forward_model_calls += 1
            current_observation = expansion_env.get_observation()

            if node.unreachable and node != self.root_node:
                print("No way expansion!")
                assert False
            child, reward = self.add_new_observation(current_observation, node, action, reward, done)
            if child is not None:
                new_nodes.append(child)
                actions_to_new_nodes.append(action)

        return new_nodes, actions_to_new_nodes

    def simulation(self, action_to_node, env):
        rewards = []
        total_steps = 0
        #  benchmarks
        times_copy = []
        times_rollouts = []
        times_steps = []
        times_randomness = []
        times_threads = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            futures = []
            paths = [None] * self.num_rollouts
            for i in range(self.num_rollouts):
                disabled_actions = []
                if self.use_disabled_actions:
                    if i < self.env.action_space.n:
                        disabled_actions.append(i)

                start = time.perf_counter()
                possible_actions = [x for x in range(self.env.action_space.n) if x not in disabled_actions]
                action_list = self.random.choice(possible_actions, self.rollout_depth)
                action_failure_probabilities = self.random.random_sample(self.rollout_depth + 1) #  +1 is for the original step
                failed_action_list = self.random.choice(possible_actions, self.rollout_depth + 1) # +1 is for the original step
                times_randomness.append(time.perf_counter() - start)

                futures.append(executor.submit(self.rollout, action_to_node, env, action_list, action_failure_probabilities, failed_action_list, i))

            start = time.perf_counter()
            for f in concurrent.futures.as_completed(futures):
                average_reward, path, i, benchmarks = f.result()
                times_copy.append(benchmarks[0])
                times_rollouts.append(benchmarks[1])
                times_steps.append(benchmarks[2])
                paths[i] = path
                rewards.append(average_reward)
                total_steps += len(path)
            times_threads.append(time.perf_counter() - start)

        self.forward_model_calls += total_steps
        return np.mean(rewards), paths, total_steps, [np.mean(times_copy), np.mean(times_rollouts), np.mean(times_steps), np.mean(times_randomness), np.mean(times_threads)]

    def rollout(self, action_to_node, env, action_list=[], action_failure_probabilities=[], failed_action_list=[], i=0):

        #  benchmark
        times_copy = None
        times_rollout = None
        times_step = []


        cum_reward = 0
        path = []
        start = time.perf_counter()
        rollout_env = deepcopy(env)
        times_copy = time.perf_counter() - start

        #rollout_env.step(action_to_node, action_failure_probabilities[0], failed_action_list[0])

        roll_start = time.perf_counter()
        previous_observation = rollout_env.get_observation()
        for idx, action in enumerate(action_list):
            start = time.perf_counter()
            state, r, done, _ = rollout_env.step(action, action_failure_probabilities[idx + 1], failed_action_list[idx + 1])
            times_step.append(time.perf_counter() - start)
            observation = rollout_env.get_observation()
            cum_reward += r

            path.append((previous_observation, observation, action, r, done))
            previous_observation = observation
            if done:
                break
        times_rollout = time.perf_counter() - roll_start

        return cum_reward, path, i, [times_copy, times_rollout, np.sum(times_step)]

    def back_propagation(self, node, reward):
        i = 0
        y = 1
        while node is not None:
            i += 1
            node.visits += 1
            node.total_value += reward * y
            node = node.parent
            y *= y


    def add_novelties_to_graph(self, paths):

        nodes = []
        node_rewards = []
        for path in paths:
            for idx, step in enumerate(path):

                observation = step[1]
                novelty = self.novelty_stats.calculate_novelty(observation)
                if self.graph.has_node(observation) is False and (novelty > 0 or self.only_add_novel_states is False):

                    for i in range(idx + 1):
                        step_i = path[i]
                        previous_observation = step_i[0]
                        current_observation = step_i[1]
                        action = step_i[2]
                        reward = step_i[3]
                        done = step_i[4]
                        novelty = self.novelty_stats.calculate_novelty(current_observation)
                        parent_node = self.graph.get_node_info(previous_observation)
                        if parent_node.unreachable and parent_node != self.root_node:
                            print("No way novelty!")
                            assert False
                        node, node_reward = self.add_new_observation(current_observation, parent_node, action, reward, done)
                        nodes.append(node)
                        node_rewards.append(node_reward)
                    if novelty >= 1:
                        node = self.graph.get_node_info(observation)
                        Logger.log_novel_data(f"Novel: {self.agent_position(node)}")
        return nodes, node_rewards

    def add_new_observation(self, current_observation, parent_node, action, reward, done):

        new_node = None

        if current_observation != parent_node.id:  # don't add node if nothing has changed in the observation
            if self.graph.has_node(current_observation) is False:  # if the node is new, create it and add to the graph
                child = Node(ID=current_observation, parent=parent_node,
                             is_leaf=True, done=done, action=action, reward=reward, visits=0,
                             novelty_value=self.novelty_stats.calculate_novelty(current_observation))
                self.add_node(child)
                new_node = child
            else:
                child = self.graph.get_node_info(current_observation)
                #if child.is_leaf: #enable for FMC optimisation, comment for full exploration
                new_node = child

            edge = self.add_edge(parent_node, child, action, reward, done)

            # TODO: Might need change for stochastic
            # revalue reward for optimal path
            if done is True:
                path, actions = self.graph.get_path(self.root_node, child)
                revalue_env = deepcopy(self.env)
                for i, action in enumerate(actions):
                    s, r, _, _ = revalue_env.step(action)
                    self.forward_model_calls += 1
                reward = r
                child.reward = reward
                edge.reward = reward

        return new_node, reward

    def get_optimal_action(self, node):

        new_root_node, action = self.select_best_step(node)
        new_root_node.chosen = True
        new_root_node.parent = None

        if self.graph.has_path(new_root_node, self.root_node):
            self.graph.reroute_path(new_root_node, self.root_node)
            self.root_node.action = self.graph.get_edge_info(self.root_node.parent, self.root_node).action

        self.root_node = new_root_node

        return action

    def set_root_node(self):

        old_root_node = self.root_node
        new_root_id = self.env.get_observation()
        self.root_node = self.graph.get_node_info(new_root_id)
        self.graph.set_root_node(self.root_node)

        if self.root_node.id != old_root_node.id:
            self.root_node.chosen = True
            self.root_node.parent = None

            # Reroute the old root node
            if self.graph.has_path(self.root_node, old_root_node):
                self.graph.reroute_path(self.root_node, old_root_node)
                old_root_node.action = self.graph.get_edge_info(old_root_node.parent, old_root_node).action

    def select_best_step(self, node, closest=False):

        if closest:
            best_node = self.graph.get_closest_done_node(only_reachable=True)
            if best_node is None:
                best_node = self.graph.get_best_node(only_reachable=True)
        else:
            best_node = self.graph.get_best_node(only_reachable=True)

        if best_node.done is True:
            self.novelty_stats.goal_found(self.steps)

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

            self.novelty_stats.update_posterior(node.id, step=self.steps)

    def add_edge(self, parent_node, child_node, action, reward, done, who="Expansion"):

        edge = Edge(ID=self.edge_counter, node_from=parent_node, node_to=child_node,
                    action=action, reward=reward, done=done)

        if not self.graph.has_edge(edge):
            self.graph.add_edge(edge)
            self.edge_counter += 1

            Logger.log_graph_data(f"{who} - New Edge: {str(self.agent_position(edge.node_from)):>12}"
                                  f" -> {str(self.agent_position(edge.node_to)):<12}"
                                  f" Action: {self.env.agent_action_mapper(edge.action):<16}")

        if child_node.unreachable is True and child_node != self.root_node:  # if child was unreachable make it reachable through this parent
            child_node.set_parent(parent_node)
            child_node.action = action
            child_node.unreachable = False

        return edge

    def get_metrics(self):

        metrics = dict(
            total_nodes=len(self.graph.graph.nodes),
            frontier_nodes=len(self.graph.frontier),
            forward_model_calls=self.forward_model_calls,
             )

        metrics.update(self.novelty_stats.get_metrics())
        return metrics
