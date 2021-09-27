import matplotlib.pyplot as plt
import yaml
import time
import datetime
import pandas as pd

from tqdm import tqdm
from Agents.MCTS.MCTSAgent import MCTSAgent

from Environments.MyMinigridEnv import MyMinigridEnv
from Environments.CustomDoorKeyEnv import CustomDoorKey
from Utils.Logger import Logger, plot_images

def load_agent_configuration(path):
    with open(path, 'r') as stream:
        return yaml.safe_load(stream)

def create_environment(env_name):
    if "Custom" in env_name:
        return CustomDoorKey(size=16)
    else:
        return MyMinigridEnv(env_name, seed=env_seed)


def run_experiment(agent_config_path, env_name, env_seed, agent_seed, verbose=True):

    agent_config = load_agent_configuration(agent_config_path)
    env = create_environment(env_name)

    Logger.setup(env_info=env.name, path=f"{env_seed}_{agent_seed}")
    agent = MCTSAgent(env, seed=agent_seed, config=agent_config, verbose=verbose)

    images = [env.render()]
    total_reward = 0
    if verbose:
        env.get_action_list()
        print(agent.info())
        plt.imshow(images[0])
        plt.show()
        plt.close()

    start_time = time.time()
    for i in range(100):
        action = agent.plan(draw_graph=False)
        state, reward, done, info = agent.act(action)
        images.append(env.render())
        total_reward += reward

        if done:
            break
    end_time = time.time()

    Logger.log_data(f"Game finished (Total nodes: {agent.novelty_stats.total_data_points})")
    Logger.close()
    agent.graph.save_graph("graph")

    plot_images(str(env_seed) + "_" + str(agent_seed), images, total_reward, verbose)

    metrics = agent.get_metrics()
    metrics.update(solved=total_reward > 0)
    metrics.update(number_of_steps=i)
    metrics.update(time_elapsed=datetime.timedelta(seconds=int(end_time - start_time)))
    return metrics


if __name__ == "__main__":

    env_name = 'MiniGrid-Empty-16x16-v0'

    # 7 easy
    # 109 medium
    # 3 medium
    # 35 hard
    # 121 very hard

    agent_seeds = range(25)
    env_seeds = range(1)
    agent_configs = [
        "AgentConfig/mcts_0.yaml",
    ]

    order_metrics = [
        'solved',
        'number_of_steps',
        'forward_model_calls',
        'key_found_nodes',
        'key_found_FMC',
        'door_found_nodes',
        'door_found_FMC',
        'goal_found_nodes',
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
                    run_experiment(agent_config, env_name, env_seed=env_seed, agent_seed=agent_seed, verbose=False)
                metrics_data_frame = pd.DataFrame(experiment_metrics, index=order_metrics).T
                Logger.save_experiment_metrics(agent_config, metrics_data_frame)
