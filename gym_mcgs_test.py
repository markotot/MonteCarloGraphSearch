import datetime
import time

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from tqdm import tqdm

from Agents.MCGS.MCGSAgent import MCGSAgent
from Gym_Environments import MinigridLevelLayouts
from Gym_Environments.AbstractGymEnv import MyDoorKeyEnv
from Utils.Logger import Logger, plot_images


# TODO: Differences to Go-Explore
#   1) Propagating novelty bonus

# TODO: BUGS -
#  1) should be fixed --- fricking OOP --- action for step is sometimes None during rollout, very rarely but can happen (16x16, episodes=5, num_rollouts=24, rollout_depth=200)
#  2) should be fixed - children_criteria is empty in self.select_child(node, criteria_"value") (16x16, episodes=5, num_rollouts=24, rollout_depth=200)
#  3) should be fixed - if something is marked as not reachable, it will never become reachable again (can be fixed, but takes a lot of computation)
#  4) should be fixed - optimize route after rollouts
#  5) should be fixed - !! important !! Action trajectory doesn't reflect the real state
#  6) should be fixed - circular parenting, infinite loop in backprop

# TODO: Improvements
#  1) done - get_optimal_action() based on the best node, not just the best child
#  2) implement softmax for select_from_frontier()
#  3) for atari we might not need deepcopy/dijkstra
#  4) try to make a summarization of the graph using loops/cliques
#  5) try a Value Function with exploration
#  6) compare with a state of the art MCTS
#  7) test the disabled actions and see if it's an improvement or not
#  8) After finding a done node, optimize the path to it first

# TODO: restrictions
#  1) node can't have edge into itself (problem with empty frontier)
#  2) stochastic environment not supported
#  3) partial observable env not supported, states need to be MDP
#  4) env isn't perfect for rollouts - more moves you make in the env, less the reward - meaning later rollouts give less reward

# TODO next:
#   make novelties be continuous rather than discrete yes/no
#   parallelize BFS
#   reroute by checking only for children

# TODO: stochastic
#   Log the currently known path
#   Log the selected action -> enacted action
#   Do continuous selection (UCB) instead of frontier
#   The Target node (best node changes due to rollouts)


def get_size_from_name(env_name):
    env_size = env_name.split("-")[2]
    return int(env_size.split("x")[0])


def load_agent_configuration(path):
    with open(path, 'r') as stream:
        return yaml.safe_load(stream)


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

    #  planning loop
    for i in range(100):
        action = agent.plan(draw_graph=True)
        state, reward, done, info = agent.act(action)

        images.append(env.render())

        # plt.imshow(images[-1])
        # plt.show()
        # plt.close()

        total_reward += reward
        if done:
            break

    end_time = time.time()

    Logger.log_data(f"Game finished (Total nodes: {agent.novelty_stats.total_data_points})")
    Logger.close()
    agent.graph.save_graph(f"Data/{env_seed}_{agent_seed}")

    if verbose:
        plot_images(str(env_seed) + "_" + str(agent_seed), images, total_reward)

    metrics = agent.get_metrics()
    metrics.update(solved=total_reward > 0)
    metrics.update(number_of_steps=i)
    metrics.update(time_elapsed=datetime.timedelta(seconds=int(end_time - start_time)))
    metrics.update(env_name=env_name)
    metrics.update(action_failure_prob=action_failure_prob)
    return metrics


if __name__ == "__main__":

    # env_name = 'MiniGrid-DoorKey-25x25-v0'
    env_name = 'MiniGrid-DoorKey-8x8-v0'
    # env_name = 'Custom-DoorKey-16x16-v0'
    # 7 easy
    # 109 medium
    # 3 medium
    # 35 hard
    # 121 very hard

    # custom_level = MinigridLevelLayouts.two_ways16
    custom_level = None
    action_failure_prob = 0.0

    agent_seeds = range(0, 1)
    # agent_seeds = [0]
    env_seeds = range(0, 1)
    agent_configs = [
        "AgentConfig/mcgs_0.yaml",
    ]

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
                Logger.save_experiment_metrics(agent_config, metrics_data_frame)
