import matplotlib.pyplot as plt
import yaml
import time
import datetime
import pandas as pd


from tqdm import tqdm
from Agents.MCGS.MCGS_QDRNAgent import MCGSAgent
from Gym_Environments import MinigridLevelLayouts
from Gym_Environments.AbstractGymEnv import MyDoorKeyEnv

#from Environments.MyMinigridEnv import MyMinigridEnv
#from Environments.CustomDoorKeyEnv import CustomDoorKey
from Utils.Logger import Logger, plot_images


def load_agent_configuration(path):
    with open(path, 'r') as stream:
        return yaml.safe_load(stream)


def get_size_from_name(env_name):
    env_size = env_name.split("-")[2]
    return int(env_size.split("x")[0])


def initialize_global_variables(agent, env, verbose=True):
  if verbose:
    print(env.get_action_list())
    print(agent.info())
    plt.imshow(env.render())
    plt.show()
    plt.close()

  total_reward, steps, done = 0, 0, False
  start_time = time.time()
  images = [env.render()]
  return total_reward, steps, done, start_time, images


def play_actions(agent, actions, total_reward, images, steps, display_output=True): 
  for a in range(0, len(actions)):
    state, reward, done, info = agent.act(actions[a])
    agent.steps += 1
    total_reward += reward
    images.append(agent.env.render())
    Logger.log_data(f"Current position: {str(agent.agent_position(agent.root_node)):<40}"
                    f"Action: {agent.env.agent_action_mapper(actions[a]):<12}")

    if display_output:
      plt.imshow(agent.env.render())
      plt.show()
      plt.close()
    if done:
      break
  return done, total_reward, steps, images


def run_experiment(agent_config_path, env_name, action_failure_prob, env_seed, agent_seed, custom_level=None,
                   verbose=True):
    agent_config = load_agent_configuration(agent_config_path)
    size = get_size_from_name(env_name=env_name)
    env = MyDoorKeyEnv(size=size, action_failure_prob=action_failure_prob, seed=env_seed, ascii=custom_level[0] if custom_level is not None else None)

    path = f"{env_seed}_{agent_seed}" if custom_level is None else f"{custom_level[1]}_{agent_seed}"
    Logger.setup(env_info=env.name, file_name=path)
    agent = MCGSAgent(env, seed=agent_seed, config=agent_config, verbose=verbose)

    images = [env.render()]
    total_reward = 0

    plt.imshow(images[0])
    plt.show()
    plt.close()
    if verbose:
        env.get_action_list()
        print(agent.info())

    start_time = time.time()

    misfired_actions = 0
    #  planning loop
    for i in range(100):
        action = agent.plan(draw_graph=False)
        state, reward, done, info = agent.act(action)
        print(action, agent.env.action)
        if action != agent.env.action:
            misfired_actions += 1

        images.append(env.render())

        plt.imshow(images[-1])
        plt.show()
        plt.close()

        total_reward += reward
        if done:
            break

    end_time = time.time()

    Logger.log_data(f"Game finished (Total nodes: {agent.novelty_stats.total_data_points})")
    Logger.close()
    #agent.graph.save_graph(f"Data/{env_seed}_{agent_seed}")

    if verbose:
        plot_images(str(env_seed) + "_" + str(agent_seed), images, total_reward)

    metrics = agent.get_metrics()
    metrics.update(solved=total_reward > 0)
    metrics.update(number_of_steps=i)
    metrics.update(time_elapsed=datetime.timedelta(seconds=int(end_time - start_time)))
    metrics.update(env_name=env_name)
    metrics.update(action_failure_prob=action_failure_prob)
    metrics.update(misfired_actions=misfired_actions)
    return metrics


if __name__ == "__main__":

    env_name = 'MiniGrid-DoorKey-16x16-v0' 
    
    agent_seeds = range(0,2)     
    env_seeds = range(0,10)
    #env_seed = 0
    
    # custom_levels = [MinigridLevelLayouts.labyrinth25] #[MinigridLevelLayouts.four_rooms16, MinigridLevelLayouts.middle16, MinigridLevelLayouts.labyrinth16, None] 
    custom_level = None
    action_failure_prob = 0.2

    agent_configs = ['AgentConfig/mcgs+qdrn_0.yaml']
    
    order_metrics = [
    'env_name',
    'action_failure_prob',
    'solved',
    'iterations',
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
    'time_elapsed',
    'misfired_actions'
    ]


    Logger.setup_experiment_folder(env_name)
    loop = tqdm(agent_configs)
    experiment_metrics = dict()
    for agent_config in loop:
        for env_seed in env_seeds:#for custom_level in custom_levels:
            for agent_seed in agent_seeds:
                loop.set_description(f"env: {env_seed} agent_seed: {agent_seed} agent_config: {agent_config}")
                experiment_metrics[
                    f"{agent_config}_{env_seed if custom_level is None else custom_level[1]}_{agent_seed}"] = \
                    run_experiment(agent_config_path=agent_config,
                                   env_name=env_name,
                                   env_seed=env_seed,
                                   action_failure_prob=action_failure_prob,
                                   agent_seed=agent_seed,
                                   custom_level=custom_level,
                                   verbose=False)

    
                metrics_data_frame = pd.DataFrame(experiment_metrics, index=order_metrics).T
                print(metrics_data_frame.mean())
                Logger.save_experiment_metrics(agent_config, metrics_data_frame)
