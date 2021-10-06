from EdouardMCGS.run_experiments import run_experiment
from Environments.MyMinigridEnv import MyMinigridEnv
from Utils.Logger import plot_images

env = MyMinigridEnv('MiniGrid-DoorKey-8x8-v0', seed=7)

agent_config = {
    "__class__": "<class 'rl_agents.agents.tree_search.graph_based.GraphBasedPlannerAgent'>",
    "gamma": 0.99,
    "budget": 1000,
}

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
