import random
from copy import deepcopy

from ribs.archives import GridArchive
from ribs.optimizers import Optimizer
from ribs.emitters import OptimizingEmitter
import pandas as pd

class Diversity:

  def __init__(self, num_features):
    self.num_features = num_features
    self.unique_obs = {i:[] for i in range(num_features)}
    #self.freq_obs = {i:{} for i in range(num_features)}
    self.freq_obs = pd.DataFrame(columns = ['feature', 'obs', 'freq'])
    self.obs_node_map = {i:{} for i in range(num_features)}
    self.reward_flag = False


  def insert_freq_row(self, f, obs):
     self.freq_obs.loc[-1] = [f, obs, 1]  # adding a row
     self.freq_obs.index = self.freq_obs.index + 1  # shifting index
     self.freq_obs = self.freq_obs.sort_index()  # sorting by index


  def add_child_to_diversity(self, child):
    obs = list(child.id)
    for f in range(self.num_features):
      if (f == 3) and obs[f] == None:
        obs[f] = 'None'
      if obs[f] not in self.unique_obs[f]:
        self.unique_obs[f].append(obs[f])
        self.insert_freq_row(f, obs[f])
        self.obs_node_map[f][obs[f]] = [child]
      else:
        self.freq_obs.loc[(self.freq_obs['feature']==f) & (self.freq_obs['obs']==obs[f]), 'freq'] += 1
        self.obs_node_map[f][obs[f]].append(child)


  def get_most_diverse_obs(self):
    most_diverse_obs = self.freq_obs.sort_values('freq').head(1).values[0]
    return most_diverse_obs


  def get_diverse_obs_nodes(self, most_diverse_obs):
    diverse_obs_nodes = []
    best = 0
    while (diverse_obs_nodes == []):
      for i in self.obs_node_map[most_diverse_obs[0]][most_diverse_obs[1]]:
        if i.unreachable==False:
          diverse_obs_nodes.append(i)
      best += 1
      if diverse_obs_nodes == []:
        most_diverse_obs = self.freq_obs.sort_values('freq').values[best]
    return diverse_obs_nodes, most_diverse_obs
  

  def check_diversity_in_obs(self, obs):
    diverse_keys = []
    for f in range(self.num_features):
      if obs[f] not in self.unique_obs[f]:
        diverse_keys.append([f, obs[f]])
    return diverse_keys



class QD_Search:

  def __init__(self, width, actions_n, agent_config):
    self.all_actions = [i for i in range(actions_n)]
    self.width = width
    self.num_of_emitters = agent_config['QDS_num_of_emitters']
    self.total_itrs = agent_config['QDS_total_itrs']
    self.batch_size = agent_config['QDS_batch_size']
    self.bounds = None
    self.initial_model = None
    self.archive = None
    self.emitters = None
    self.optimizer = None
    self.archive_df = None
    self.rollout_depth = agent_config['QDS_rollout_depth']
    self.qd_budget = agent_config['QDS_budget_per_step']


  def initialize_qd(self, diversity_path_actions):
    self.initial_model = [random.choice(self.all_actions) for _ in range(self.rollout_depth)] #diversity_path_actions
    self.bounds = [(self.all_actions[0], self.all_actions[-1]) for i in range(len(self.initial_model))]
    self.archive = GridArchive([self.width, self.width], 
                               [(0, self.width), (0, self.width)])
    self.emitters = [OptimizingEmitter(self.archive, self.initial_model, 1, batch_size=self.batch_size, 
                                       bounds = self.bounds) for _ in range(self.num_of_emitters)] 
    self.optimizer = Optimizer(self.archive, self.emitters)


  def rollout(self, env, solution, agent):
    total_reward = 0
    path = []
    actions = [int(a) for a in solution]

    simulate_env = deepcopy(env)
    previous_observation = simulate_env.get_observation()
    
    for action in actions:
        state, r, done, _ = simulate_env.step(action)
        agent.forward_model_calls += 1
        #agent.remaining_budget -= 1

        observation = simulate_env.get_observation()
        #agent.diversity.add_child_to_diversity(agent.graph.get_node_info(observation))      
        total_reward += r
        path.append((previous_observation, observation, action, r, done))
        previous_observation = observation
        if done:
            break

    _, _ = agent.add_novelties_to_graph([path])
    return path[-1][1], path, total_reward


  def get_scores(self, agent, path, most_diverse_obs, total_reward):
    diverse_keys = []
    diverse_obs_states = 0
    for p in path:
      observation = p[1]
      
      if agent.graph.has_node(observation) == False:
        diverse_keys = agent.diversity.check_diversity_in_obs(observation)

      if observation[most_diverse_obs[0]] == most_diverse_obs[1]:
        diverse_obs_states += 1
    return diverse_obs_states + len(diverse_keys) + total_reward*10


  def qd_optimize(self, agent, most_diverse_obs):
    len_sol = len(self.initial_model)
    budget = self.qd_budget + agent.remaining_budget
    fmc = 0
    for itr in range(1, self.total_itrs + 1):
      solutions = self.optimizer.ask()
      objs, bcs = [], []    
      for i in range(len(solutions)):
        fmc += len_sol

        last_step, path, total_reward = self.rollout(agent.env, solutions[i], agent)
        diversity_score = self.get_scores(agent, path, most_diverse_obs, total_reward)

        objs.append(diversity_score)
        bcs.append([last_step[0], last_step[1]]) 
      self.optimizer.tell(objs, bcs)
      if fmc > budget:
        break
    self.archive_df = self.archive.as_pandas().astype(int)


  def qd_search(self, agent, most_diverse_obs, diversity_path_actions):
    self.initialize_qd(diversity_path_actions)
    self.qd_optimize(agent, most_diverse_obs)


  def get_qd_elite(self):
    solutions = []
    sol_cols = ['solution_'+str(i) for i in range(self.archive_df.shape[1]-5)]
    for i in self.archive_df.loc[self.archive_df.objective==self.archive_df.objective.max()].index:
      solutions.append(self.archive_df.loc[i, sol_cols].values)
    actions = random.choice(solutions)
    return actions
