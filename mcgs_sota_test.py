from EdouardMCGS.run_experiments import run_experiment
from Environments.MyMinigridEnv import MyMinigridEnv
from Gym_Environments.AbstractGymEnv import MyDoorKeyEnv
from Utils.Logger import plot_images

env_seed = 18 # last done was 15
# env = MyMinigridEnv('MiniGrid-DoorKey-16x16-v0', action_failure_prob=0.2)
env = MyDoorKeyEnv(size=25, action_failure_prob=0, seed=env_seed)
plot_images(1, [env.render()], 0, True)
# agent_type = "<class 'rl_agents.agents.tree_search.graph_based_stochastic.StochasticGraphBasedPlannerAgent'>"
agent_type = "<class 'rl_agents.agents.tree_search.graph_based.GraphBasedPlannerAgent'>"


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

print(env_seed)
_, images_per_episode, agent, env = run_experiment(env, agent_config, options)
for images in images_per_episode:
    plot_images(len(images), images, 0, True)
