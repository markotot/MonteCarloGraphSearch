import datetime
import os
from pathlib import Path
import gym
import json
from docopt import docopt
from itertools import product
from multiprocessing.pool import Pool
from gym_minigrid.wrappers import *

from rl_agents.trainer import logger
from EdouardMCGS.evaluation import Evaluation
from rl_agents.agents.common.factory import load_agent, load_environment
from rl_agents.agents.tree_search.graph_based import GraphNode
from Environments.MiniGridEnv import MiniGridEnv
from Utils.Logger import plot_images

BENCHMARK_FILE = "benchmark_summary"
LOGGING_CONFIG = "configs/logging.json"
VERBOSE_CONFIG = "configs/verbose.json"


def evaluate(env, agent_config, options):
    """
        Evaluate an agent interacting with an environment.
    :param environment_config: the path of the environment configuration file
    :param agent_config: the path of the agent configuration file
    :param options: the evaluation options
    """

    #  to make it work with our MiniGridEnv
    env.unwrapped = env.env.unwrapped
    env.observation_space = env.env.observation_space
    env.reward_range = env.env.reward_range
    env.metadata = env.env.metadata
    env.spec = env.env.spec
    env.seed = env.env.seed
    env.render = env.env.render
    env.close = env.env.close
    #  to make it work with our MiniGridEnv

    logger.configure()
    # if options['--verbose']:
    #     logger.configure(VERBOSE_CONFIG)
    agent = load_agent(agent_config, env)
    run_directory = None
    # if options['--name-from-config']:
    #     run_directory = "{}_{}_{}".format(Path(agent_config).with_suffix('').name,
    #                               datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
    #                               os.getpid())
    options["--seed"] = (
        int(options["--seed"]) if options["--seed"] is not None else None
    )
    evaluation = Evaluation(
        env,
        agent,
        run_directory=run_directory,
        num_episodes=int(options["--episodes"]),
        sim_seed=options["--seed"],
        recover=False,  # options['--recover'] or options['--recover-from'],
        display_env=not options["--no-display"],
        display_agent=not options["--no-display"],
        display_rewards=not options["--no-display"],
    )

    images_per_episode = evaluation.train()
    print(GraphNode.forward_model_calls)
    return os.path.relpath(evaluation.monitor.directory), images_per_episode, agent, env



env = MiniGridEnv('MiniGrid-DoorKey-16x16-v0')

agent_config = {
    "__class__": "<class 'rl_agents.agents.tree_search.graph_based.GraphBasedPlannerAgent'>",
    "gamma": 0.99,
    "budget": 8000,
}
options = {
    "--seed": 35,
    "--no-display": True,
    "--episodes": 1,
    "--train": True,
    "--test": True,
    "--processed": 16
}

_, images_per_episode, agent, env = evaluate(env, agent_config, options)
for images in images_per_episode:
    plot_images(len(images), images, 0, True)

#%%
import networkx as nx

graph = nx.DiGraph()

for node in agent.planner.nodes.values():
    graph.add_node(node, info=node)

for node in agent.planner.nodes.values():
    for child in node.children.values():
        if not graph.has_edge(node, child):
            graph.add_edge(node, child)

print(len(graph.nodes))

# draw_graph(agent.planner.root, graph)