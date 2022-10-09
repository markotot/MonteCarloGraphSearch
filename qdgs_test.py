import matplotlib.pyplot as plt
import yaml
import time
import datetime
import pandas as pd


from tqdm import tqdm
from Agents.MCGS.QDGSAgent import MCGSAgent

from Environments.MyMinigridEnv import MyMinigridEnv
from Environments.CustomDoorKeyEnv import CustomDoorKey
from Utils.Logger import Logger, plot_images


def load_agent_configuration(path):
    with open(path, 'r') as stream:
        return yaml.safe_load(stream)


def create_environment(env_name, action_failure_prob, env_seed):
    if "Custom" in env_name:
        return CustomDoorKey(size=16)
    else:
        return MyMinigridEnv(env_name, action_failure_prob=action_failure_prob, seed=env_seed)


def initialize_global_variables(agent, env, verbose=True):
  if verbose:
    print(env.get_action_list())
    print(agent.info())
    plt.imshow(env.render())
    plt.show()
    plt.close()

  total_reward, steps, done = 0, 0, False
  novelty_keys = []
  start_time = time.time()
  images = [env.render()]
  return total_reward, steps, done, novelty_keys, start_time, images


def play_actions(agent, actions, total_reward, images, steps, display_output=True): 
  for a in range(0, len(actions)):
    state, reward, done, info = agent.act(actions[a])
    steps += 1
    observation = agent.env.get_observation()
    agent.check_metrics(observation, steps)
    total_reward += reward
    images.append(agent.env.render())
    Logger.log_data(f"Current position: {str(agent.agent_position(agent.root_node)):<40}"
                    f"Action: {agent.env.agent_action_mapper(actions[a]):<12}")

    if display_output:
      plt.imshow(agent.env.render())
      plt.show()
      plt.close()
    if done:
      agent.goal_found = (agent.node_counter, steps, agent.forward_model_calls)
      break
  return done, total_reward, steps, images


def run_experiment(agent_config_path, env_name, action_failure_prob, env_seed, agent_seed, verbose=True):

    agent_config = load_agent_configuration(agent_config_path)
    env = create_environment(env_name=env_name, action_failure_prob=action_failure_prob, env_seed=env_seed)

    Logger.setup(env_info=env.name, path=f"{env_seed}_{agent_seed}")
    agent = MCGSAgent(env, seed=agent_seed, config=agent_config, verbose=verbose)

    total_reward, steps, done, novelty_keys, start_time, images = initialize_global_variables(agent, env, verbose=True)

    for itr in range(agent_config['no_of_itr']):
    
      # MCGS & QDS
      actions = agent.plan(novelty_keys, draw_graph=agent_config['display_graph'])
      # Play
      done, total_reward, steps, images = play_actions(agent, actions, total_reward, images, steps, display_output=agent_config['display_output'])
      if done:
        break
    
    end_time = time.time()
    Logger.log_data(f"Game finished (Total nodes: {agent.node_counter})")
    Logger.close()
    agent.graph.save_graph("graph")
    plot_images(str(env_seed) + "_" + str(agent_seed), images, total_reward, verbose)

    metrics = agent.get_metrics()
    metrics.update(solved=total_reward > 0)
    metrics.update(number_of_steps=steps)
    metrics.update(time_elapsed=datetime.timedelta(seconds=int(end_time - start_time)))
    metrics.update(env_name=env_name)
    metrics.update(action_failure_prob=action_failure_prob)
    return metrics


if __name__ == "__main__":

    env_name = 'MiniGrid-DoorKey-16x16-v0' 
    action_failure_prob = 0.0
    
    agent_seeds = range(0,2)     
    env_seeds = range(0,20)


    #'AgentConfig/qdgs_0.yaml', 'AgentConfig/qdgs_2.yaml', 'AgentConfig/qdgs_3.yaml',
    agent_configs = ['AgentConfig/qdgs_4.yaml',  'AgentConfig/qdgs_1.yaml']
    
    order_metrics = [
    'env_name',
    'action_failure_prob',
    'solved',
    'number_of_steps',
    'forward_model_calls',
    'key_found_nodes',
    'key_found_steps',
    'key_found_FMC',
    'door_found_nodes',
    'door_found_steps',
    'door_found_FMC',
    'goal_found_nodes',
    'goal_found_steps',
    'goal_found_FMC',
    'total_nodes',
    'frontier_nodes',
    'time_elapsed'
    ]


    Logger.setup_experiment_folder(env_name)
    loop = tqdm(agent_configs)
    experiment_metrics = dict()
    for agent_config in loop:
        for env_seed in env_seeds:
            for agent_seed in agent_seeds:
                loop.set_description(f"env: {env_seed} agent_seed: {agent_seed} agent_config: {agent_config}")
                experiment_metrics[f"{agent_config}_{env_seed}_{agent_seed}"] = \
                    run_experiment(agent_config_path=agent_config,
                                   env_name=env_name,
                                   env_seed=env_seed,
                                   action_failure_prob=action_failure_prob,
                                   agent_seed=agent_seed,
                                   verbose=True)
    
                metrics_data_frame = pd.DataFrame(experiment_metrics, index=order_metrics).T
                Logger.save_experiment_metrics(agent_config, metrics_data_frame)
