from Agents.AbstractAgent import AbstractAgent
from Agents.MCGS.Graph import Graph

from QualityDiversity.QDRFN import Diversity
from QualityDiversity.QDRFN import QD_Search
import matplotlib.pyplot as plt


from Utils.Logger import Logger
import numpy as np
import concurrent.futures
from math import sqrt, log
from copy import deepcopy
from tqdm import trange
import random

rollout_executor = concurrent.futures.ProcessPoolExecutor(max_workers=16)


class MCGSAgent(AbstractAgent):

    config = None

    def __init__(self, env, seed, config, verbose):
        super().__init__(env=env, seed=seed, verbose=verbose)
        MCGSAgent.config = config

        self.diversity = Diversity(6)
        self.qds = QD_Search(self.env.width, self.env.action_space.n, config)

        self.graph = Graph(seed, config)
        self.node_counter = 0
        self.edge_counter = 0
        self.steps = 0
        self.iterations = 0

        self.episodes = config['MCGS_episodes']
        self.num_rollouts = config['MCGS_num_rollouts']
        self.rollout_depth = config['MCGS_rollout_depth']
        self.budget_per_step = config['MCGS_budget_per_step']
        self.remaining_budget = self.budget_per_step

        self.use_novelty = config['MCGS_use_novelty']
        self.novelty_factor = config['MCGS_novelty_factor']

        self.noisy_frontier_selection = config['MCGS_noisy_frontier_selection']
        self.add_nodes_during_rollout = config['MCGS_add_nodes_during_rollout']
        self.use_back_propagation = config['MCGS_use_back_propagation']
        self.use_disabled_actions = config['MCGS_use_disabled_actions']
        self.only_add_novel_states = config['MCGS_only_add_novel_states']

        self.root_node = Node(ID=self.env.get_observation(), parent=None, is_leaf=True, done=False,
                              action=None, reward=0, visits=0, novelty_value=0)
        self.root_node.chosen = True
        self.node_counter += 1
        self.num_sim = 0
        self.add_node(self.root_node)
        self.mcgs_forward_model_calls = 0
        self.qds_forward_model_calls = 0
        self.useless_forward_model_calls = 0
        
        Logger.log_data(self.info(), time=False)
        Logger.log_data(f"Start: {str(self.agent_position(self.root_node))}")
        self.start_node = self.root_node
        
        self.key_subgoal = (-1, -1, -1)
        self.door_subgoal = (-1, -1, -1)
        self.goal_found = (-1, -1, -1)


    def internal_plan(self, draw_graph):
                
        internal_env = deepcopy(self.env)
        for i in range(100):
            actions = self.plan(internal_env, draw_graph)
            # Play
            done = self.play_actions(internal_env, actions, display_output=True)
            self.iterations += 1

            if done:
              break
        
        observation = internal_env.get_observation()
        final_node = self.graph.get_node_info(observation)
        start_observation = self.env.get_observation()   
        start_node = self.graph.get_node_info(start_observation)
        
        actions = self.graph.get_path(start_node, final_node)[1]   
        
        return actions
        

    def plan(self, internal_env, draw_graph=True) -> int:
        self.remaining_budget = self.budget_per_step
        self.set_root_node(internal_env)
        self.graph.reroute_all()

        budget_per_iteration = 0
        while self.remaining_budget > budget_per_iteration:
            previous_remaining_budget = self.remaining_budget

            selection_env = deepcopy(internal_env)
            node = self.selection(selection_env)
            children, actions_to_children = self.expansion(node, selection_env)

            for idx in range(len(children)):
                child_average_reward, novelties = self.sequential_simulation(actions_to_children[idx], selection_env)

                if self.add_nodes_during_rollout:
                    rollout_nodes, rewards = self.add_novelties_to_graph(novelties)

            budget_per_iteration = previous_remaining_budget - self.remaining_budget
        
        reward_flag = self.check_reward_reached()

        if not reward_flag:
          # QD Search
          self.qds.qd_search(self, internal_env)
        reward_flag = self.check_reward_reached()
          
        # Decide actions
        if reward_flag:
          actions, next_node = self.get_reward_action()
        else:
          actions, next_node = self.get_quickest_diversity_node_actions()
        
        if draw_graph:
            self.graph.draw_graph()
                    
        return actions


    def play_actions(self, internal_env, actions, display_output=True): 
        path = []
        previous_observation = internal_env.get_observation()
        
        for a in range(0, len(actions)):
            state, reward, done, info = internal_env.step(actions[a])
            observation = internal_env.get_observation()
            path.append((previous_observation, observation, actions[a], reward, done))
            previous_observation = observation
        
            if display_output:
              plt.imshow(internal_env.render())
              plt.show()
              plt.close()
            if done:
              break
       
        _, _ = self.add_novelties_to_graph([path])
        return done


    def get_quickest_diversity_node_actions(self):
      diversity_obs_nodes, most_diverse_obs = self.diversity.get_diverse_obs_nodes()
      diversity_node = random.choice([n for n in diversity_obs_nodes if n.unreachable is False])
      
      actions = self.graph.get_path(self.root_node, diversity_node)[1]   
      
      return actions, diversity_node


    def check_reward_reached(self):
      for n in self.graph.get_all_nodes_info():
        if n.total_value>0:
          return True 


    def selection(self, env):

        if self.root_node.is_leaf:
            return self.root_node

        selectable_nodes = [x for x in self.graph.frontier if x.unreachable is False]
        
        if len(selectable_nodes) == 0:
            return self.root_node
        else:
            most_diverse_feature = self.diversity.freq_obs.loc[np.argmin(self.diversity.freq_obs.freq)]
            diverse_nodes = [n for n in selectable_nodes if n.id[most_diverse_feature['feature']] == most_diverse_feature['obs']]
            
            if len(diverse_nodes) == 0:
                node = random.choice(selectable_nodes)
            else:
                node = random.choice(diverse_nodes)
            
            assert self.graph.has_path(self.root_node, node)        
            selected_node = self.go_to_node(node, env)
            return selected_node


    def go_to_node(self, destination_node, env):

        observation = env.get_observation()
        node = self.graph.get_node_info(observation)

        reached_destination = False
        while self.graph.has_path(node, destination_node) and not reached_destination:
            observations, actions = self.graph.get_path(node, destination_node)
            for idx, action in enumerate(actions):
                previous_observation = env.get_observation()
                parent_node = self.graph.get_node_info(previous_observation)

                state, reward, done, _ = env.step(action)
                self.useless_forward_model_calls += 1
                self.remaining_budget -= 1

                current_observation = env.get_observation()
                if not self.graph.has_node(current_observation):
                    self.add_new_observation(current_observation, parent_node, action, reward, done)
                elif not self.graph.has_edge_by_nodes(parent_node, self.graph.get_node_info(current_observation)):
                    self.add_edge(parent_node, self.graph.get_node_info(current_observation), action, reward, done)

                if observations[idx + 1] != current_observation:
                    node = self.graph.get_node_info(current_observation)
                    break
                if destination_node.id == env.get_observation():
                    reached_destination = True
                    break

        return self.graph.get_node_info(env.get_observation())


    def expansion(self, node, env):

        if node.done:
            return [], []

        new_nodes = []
        actions_to_new_nodes = []
        if node.is_leaf:
            node.is_leaf = False
            self.graph.remove_from_frontier(node)

        for action in range(self.env.action_space.n):

            expansion_env = deepcopy(env)
            state, reward, done, _ = expansion_env.step(action)
            self.useless_forward_model_calls += 1
            self.remaining_budget -= 1
            current_observation = expansion_env.get_observation()
            self.check_metrics(current_observation, reward, done, 'Expansion')

            child, reward = self.add_new_observation(current_observation, node, action, reward, done)
            if child is not None:
                new_nodes.append(child)
                actions_to_new_nodes.append(action)

        return new_nodes, actions_to_new_nodes


    def sequential_simulation(self, action_to_node, env):
        rewards = []

        paths = []

        for i in range(self.num_rollouts):
            
            disabled_actions = []
            if self.use_disabled_actions:
                if i < self.env.action_space.n:
                    disabled_actions.append(i)

            possible_actions = [x for x in range(self.env.action_space.n) if x not in disabled_actions]
            action_list = self.random.choice(possible_actions, self.rollout_depth)
            action_failure_probabilities = self.random.random_sample(self.rollout_depth + 1)  # +1 is for the original step
            failed_action_list = self.random.choice(possible_actions, self.rollout_depth + 1)  # +1 is for the original step

            average_reward, path = self.rollout(action_to_node, env, action_list, action_failure_probabilities, failed_action_list)
            paths.append(path)
            rewards.append(average_reward)

        return np.mean(rewards), paths


    def rollout(self, action_to_node, env, action_list=[], action_failure_probabilities=[], failed_action_list=[]):

        cum_reward = 0
        path = []
        rollout_env = deepcopy(env)
        rollout_env.stochastic_step(action_to_node, action_failure_probabilities[0], failed_action_list[0])
        self.mcgs_forward_model_calls += 1
        self.remaining_budget -= 1

        previous_observation = rollout_env.get_observation()
        for idx, action in enumerate(action_list):
            state, r, done, _ = rollout_env.stochastic_step(action, action_failure_probabilities[idx + 1], failed_action_list[idx + 1])
            observation = rollout_env.get_observation()
            self.mcgs_forward_model_calls += 1
            self.remaining_budget -= 1
            self.check_metrics(observation, r, done, 'mcgs_rollout')

            cum_reward += r
            path.append((previous_observation, observation, action, r, done))
            previous_observation = observation
            if done:
                break

        return cum_reward, path


    def add_novelties_to_graph(self, paths):

        nodes = []
        node_rewards = []
        for path in paths:
            for idx, step in enumerate(path):

                observation = step[1]
                if self.graph.has_node(observation) is False and (self.only_add_novel_states is False):

                    for i in range(idx + 1):
                        step_i = path[i]
                        previous_observation = step_i[0]
                        current_observation = step_i[1]
                        action = step_i[2]
                        reward = step_i[3]
                        done = step_i[4]
                        parent_node = self.graph.get_node_info(previous_observation)
                        if parent_node.unreachable and parent_node != self.root_node:
                            print("No way novelty!")
                            assert False
                        node, node_reward = self.add_new_observation(current_observation, parent_node, action, reward, done)
                        nodes.append(node)
                        node_rewards.append(node_reward)
                node_to_update = self.graph.get_node_info(observation)
                self.diversity.add_child_to_diversity(node_to_update)
        return nodes, node_rewards


    def add_new_observation(self, current_observation, parent_node, action, reward, done):

        new_node = None

        if current_observation != parent_node.id:  # don't add node if nothing has changed in the observation
            if self.graph.has_node(current_observation) is False:  # if the node is new, create it and add to the graph
                child = Node(ID=current_observation, parent=parent_node,
                             is_leaf=True, done=done, action=action, reward=reward, visits=0, novelty_value=0)
                self.add_node(child)
                new_node = child
            else:
                child = self.graph.get_node_info(current_observation)
                if child.is_leaf: #enable for FMC optimisation, comment for full exploration
                  new_node = child

            _ = self.add_edge(parent_node, child, action, reward, done)
            self.diversity.add_child_to_diversity(child)
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

    def set_root_node(self, internal_env):

        old_root_node = self.root_node
        new_root_id = internal_env.get_observation()
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

        best_node = None
        if closest:
            best_node = self.graph.get_closest_done_node(only_reachable=True)

        if best_node is None:
            best_node = self.graph.get_best_node(only_reachable=True)

        if best_node is None:
            return self.root_node, 6  # if there is no reachable node, use root (6 - no action)

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
        mcgs_rollouts = "Num MCGS rollouts: " + str(self.num_rollouts)
        mcgs_depth = "MCGS Depth: " + str(self.rollout_depth)
        qds_depth = "QDS Depth: " + str(self.qds.rollout_depth)
        seed = "Seed: " + str(self.env.seed)
        return [env_name, seed, episodes, mcgs_rollouts, mcgs_depth, qds_depth]


    def add_node(self, node):

        if not self.graph.has_node(node):
            self.graph.add_node(node)

            if node.done is False:
                self.graph.add_to_frontier(node)
            self.node_counter += 1


    def add_edge(self, parent_node, child_node, action, reward, done):

        edge = Edge(ID=self.edge_counter, node_from=parent_node, node_to=child_node,
                    action=action, reward=reward, done=done)

        if not self.graph.has_edge(edge):
            self.graph.add_edge(edge)
            self.edge_counter += 1

        if child_node.unreachable is True and child_node != self.root_node:  # if child was unreachable make it reachable through this parent
            child_node.set_parent(parent_node)
            child_node.action = action
            child_node.unreachable = False

        return edge


    def check_metrics(self, obs, r, done, location):
        if obs[3] == 'key':
            if self.key_subgoal == (-1, -1, -1):
                self.key_subgoal = (self.node_counter, self.iterations+1, self.mcgs_forward_model_calls+self.useless_forward_model_calls+self.qds_forward_model_calls)
                print('Key found in ', location)
        if obs[4] == True:
            if self.door_subgoal == (-1, -1, -1):
                self.door_subgoal = (self.node_counter, self.iterations+1, self.mcgs_forward_model_calls+self.useless_forward_model_calls+self.qds_forward_model_calls)
                print('Door opened via ', location)
        if (done) | (r > 0):
            if self.goal_found == (-1, -1, -1):
                self.goal_found = (self.node_counter, self.iterations+1, self.mcgs_forward_model_calls+self.useless_forward_model_calls+self.qds_forward_model_calls)    
                print('Goal found in ', location)


    def get_metrics(self):
        
        print("\n\nMCGS simulation", self.mcgs_forward_model_calls, "goto stuff", self.useless_forward_model_calls, "QDS simulation", self.qds_forward_model_calls)

        metrics = dict(
            total_nodes=self.node_counter,
            frontier_nodes=len(self.graph.frontier),
            forward_model_calls=self.mcgs_forward_model_calls+self.useless_forward_model_calls+self.qds_forward_model_calls,
            key_found_nodes=self.key_subgoal[0],
            key_found_steps=self.key_subgoal[1],
            key_found_FMC=self.key_subgoal[2],
            door_found_nodes=self.door_subgoal[0],
            door_found_steps=self.door_subgoal[1],
            door_found_FMC=self.door_subgoal[2],
            goal_found_nodes=self.goal_found[0],
            goal_found_steps=self.goal_found[1],
            goal_found_FMC=self.goal_found[2],
             )
        return metrics


    def get_reward_action(self):
        min_len = 9999
        for n in self.graph.get_all_nodes_info():
            if (n.total_value>0) & (n.unreachable==False):
                if self.graph.has_path(self.root_node, n):
                    reward_len = self.graph.get_path_length(self.root_node, n)
                    if reward_len < min_len:
                        min_len = reward_len
                        reward_node = n
        actions = self.graph.get_path(self.root_node, reward_node)[1]
        return actions, reward_node




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
        self.novelty_value = novelty_value

        self.chosen = False
        self.unreachable = False

    def uct_value(self):
        ucb = sqrt(log(self.parent.visits + 1) / self.visits)
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