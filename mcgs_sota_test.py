from EdouardMCGS.run_experiments import run_experiment
from Environments.MyMinigridEnv import MyMinigridEnv
from Utils.Logger import plot_images

env = MyMinigridEnv('MiniGrid-Empty-16x16-v0')

#agent_type = "<class 'rl_agents.agents.tree_search.graph_based_stochastic.StochasticGraphBasedPlannerAgent'>"
agent_type = "<class 'rl_agents.agents.tree_search.graph_based.GraphBasedPlannerAgent'>"

env_seed = 24 # last done was 15
budget = 10 * 16 * 50

agent_config = {
    "__class__": agent_type,
    "gamma": 0.99,
    "budget": budget,
    "max_next_states_count": 5,
}
options = {
    "--seed": env_seed,
    "--no-display": True,
    "--episodes": 1,
    "--train": True,
    "--test": True,
    "--processes": 16
}

_, images_per_episode, agent, env = run_experiment(env, agent_config, options)
for images in images_per_episode:
    plot_images(len(images), images, 0, True)
