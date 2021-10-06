import matplotlib.pyplot as plt
import yaml
import time
import datetime
import pandas as pd

from tqdm import tqdm
from Agents.MCGS.MCGSAgent import MCGSAgent


from Environments.MiniGridEnv import MiniGridEnv
from Utils.Logger import Logger, plot_images

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

# TODO: restrictions
#  1) node can't have edge into itself (problem with empty frontier)
#  2) stochastic environment not supported/tested
#  3) partial observable env not supported, states need to be MDP
#  4) env isn't perfect for rollouts - more moves you make in the env, less the reward - meaning later rollouts give less reward

# TODO next:
#   make novelties be continuous rather than discrete yes/no (DONE)
#   parallelize BFS
#   check only for children


def run_experiment(agent_config_path, env_name, seed, verbose=True):

    with open(agent_config_path, 'r') as stream:
        agent_config = yaml.safe_load(stream)

    env = MiniGridEnv(env_name, seed=seed)
    Logger.setup(env_info=env.name, path=str(seed))

    agent = MCGSAgent(env, seed=seed, config=agent_config, verbose=verbose)
    images = [env.render()]
    total_reward = 0

    if verbose:
        env.get_action_list()
        print(agent.info())
        plt.imshow(images[0])
        plt.show()

    start_time = time.time()
    for i in range(100):
        action = agent.plan(draw_graph=False)
        state, reward, done, info = agent.act(action)
        images.append(env.render())
        total_reward += reward

        if done:
            break
    end_time = time.time()

    Logger.log_data(f"Game finished (Total nodes: {agent.state_database.total_data_points})")
    Logger.close()
    agent.graph.save_graph("graph")

    plot_images(seed, images, total_reward, verbose)

    metrics = agent.get_metrics()
    metrics.update(solved=total_reward > 0)
    metrics.update(number_of_steps=i)
    metrics.update(time_elapsed=datetime.timedelta(seconds=int(end_time - start_time)))
    return metrics


if __name__ == "__main__":

    env_name = 'MiniGrid-Empty-8x8-v0'

    # 7 easy
    # 109 medium
    # 3 medium
    # 35 hard
    # 121 very hard
    seeds = [7]
    experiments = [
        #"AgentConfig/mcgs_0.yaml",
        #"AgentConfig/mcgs_1.yaml",
        "AgentConfig/mcgs_2.yaml",
        #"AgentConfig/mcgs_3.yaml",
        #"AgentConfig/mcgs_4.yaml",
        #"AgentConfig/mcgs_5.yaml",
        #"AgentConfig/mcgs_6.yaml",
        #"AgentConfig/mcgs_7.yaml",

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
            loop.set_description(f"Doing seed {seed} for experiment {experiment}")
            experiment_metrics[experiment + "_" + str(seed)] = \
                run_experiment(experiment, env_name, seed=seed, verbose=True)
            metrics_data_frame = pd.DataFrame(experiment_metrics, index=order_metrics).T
            Logger.save_experiment_metrics(experiment, metrics_data_frame)

