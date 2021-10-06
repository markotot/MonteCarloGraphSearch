#TODO:
# Check if FMC is working
# Create .csv for metrics
# Create different agent seeds/env seeds

from EdouardMCGS.run_experiments import run_experiment
from Environments.MyMinigridEnv import MyMinigridEnv
from Utils.Logger import plot_images

agent_config = {
    "__class__": "<class 'rl_agents.agents.tree_search.graph_based_stochastic.StochasticGraphBasedPlannerAgent'>",
    "gamma": 0.99,
    "budget": 10*16*50,
    "max_next_states_count": 4,
}


for i in range(0, 1):
    env = MyMinigridEnv('MiniGrid-DoorKey-16x16-v0')

    options = {
        "--seed": 7,
        "--no-display": True,
        "--episodes": 1,
        "--train": True,
        "--test": True,
        "--processed": 16
    }

    _, images_per_episode, agent, env = run_experiment(env, agent_config, options)
    for images in images_per_episode:
        plot_images(len(images), images, 0, True)