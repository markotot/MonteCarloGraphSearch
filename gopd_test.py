from EdouardMCGS.run_experiments import run_experiment
from Environments.MyMinigridEnv import MyMinigridEnv
from Utils.Logger import plot_images

env = MyMinigridEnv('MiniGrid-DoorKey-8x8-v0')

agent_config = {
    "__class__": "<class 'rl_agents.agents.tree_search.graph_based.GraphBasedPlannerAgent'>",
    "gamma": 0.99,
    "budget": 80,
}

options = {
    "--seed": 42,
    "--no-display": True,
    "--episodes": 1,
    "--train": True,
    "--test": True,
    "--processed": 16
}

_, images_per_episode, agent, env, metrics = run_experiment(env, agent_config, options)
for images in images_per_episode:
    plot_images(len(images), images, 0, True)

#%%
"""
import networkx as nx

graph = nx.DiGraph()

for node in agent.planner.nodes.values():
    graph.add_node(node, info=node)

for node in agent.planner.nodes.values():
    for child in node.children.values():
        if not graph.has_edge(node, child):
            graph.add_edge(node, child)

print(len(graph.nodes))
"""