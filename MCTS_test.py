import matplotlib.pyplot as plt
import yaml
import time
import datetime
import pandas as pd

from tqdm import tqdm
from Agents.MCTS.MCTSAgent import MCTSAgent

from Environments.MiniGridEnv import MiniGridEnv
from Utils.Logger import Logger, plot_images


def run_experiment(agent_config_path, env_name, seed, verbose=True):

    with open(agent_config_path, 'r') as stream:
        agent_config = yaml.safe_load(stream)

    env = MiniGridEnv(env_name, seed=seed)
    Logger.setup(env_info=env.name, path=str(seed))

    agent = MCTSAgent(env, seed=seed, config=agent_config, verbose=verbose)
    images = [env.render()]
    total_reward = 0

    if verbose:
        env.get_action_list()
        print(agent.info())
        plt.imshow(images[0])
        plt.show()

    start_time = time.time()
    for i in range(100):
        action = agent.plan(draw_graph=True)
        state, reward, done, info = agent.act(action)

        print(env.get_agent_position())
        images.append(env.render())
        total_reward += reward

        if done:
            break
    end_time = time.time()

    Logger.log_data(f"Game finished (Total nodes: {agent.state_database.total_data_points})")
    Logger.close()
    agent.graph.save_graph("graph")

    plot_images(seed, images, total_reward, verbose)

    metrics = []
#    metrics.update(solved=total_reward > 0)
#    metrics.update(number_of_steps=i)
#    metrics.update(time_elapsed=datetime.timedelta(seconds=int(end_time - start_time)))
    return metrics


if __name__ == "__main__":

    env_name = 'MiniGrid-DoorKey-6x6-v0'

    # 7 easy
    # 109 medium
    # 3 medium
    # 35 hard
    # 121 very hard
    seeds = [7]
    experiments = [
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
    loop = tqdm(experiments)
    experiment_metrics = dict()
    for experiment in loop:
        for seed in seeds:
            #loop.set_description(f"Doing seed {seed} for experiment {experiment}")
            experiment_metrics[experiment + "_" + str(seed)] = \
                run_experiment(experiment, env_name, seed=seed, verbose=True)
            #metrics_data_frame = pd.DataFrame(experiment_metrics, index=order_metrics).T
            #Logger.save_experiment_metrics(experiment, metrics_data_frame)

