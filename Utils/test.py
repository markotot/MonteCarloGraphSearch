import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from Utils.Logger import Logger
from networkx.drawing.nx_agraph import graphviz_layout
from Agents.MCGS.StateDatabase import StateDatabase
from Agents.MCGS.Graph import Graph
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import matplotlib.pyplot as plt
import numpy as np
from Agents.MCTS.MCTSAgent import MCTSAgent
from Agents.MCGS.MCGSAgent import MCGSAgent
from Environments.MiniGridEnv import MiniGridEnv
from Utils.Logger import Logger, plot_images
import time

default_ascii = [
    ['Wall', 'Wall', 'Wall', 'Wall', 'Wall', 'Wall', 'Wall', 'Wall'],
    ['Wall', 'Player', ' ', 'Wall', ' ', ' ', ' ', 'Wall'],
    ['Wall', 'Key', ' ', 'Wall', ' ', ' ', ' ', 'Wall'],
    ['Wall', 'Wall', 'Door', 'Wall', ' ', ' ', ' ', 'Wall'],
    ['Wall', ' ', ' ', ' ', ' ', ' ', ' ', 'Wall'],
    ['Wall', ' ', ' ', ' ', ' ', ' ', ' ', 'Wall'],
    ['Wall', ' ', ' ', ' ', ' ', 'Goal', ' ', 'Wall'],
    ['Wall', 'Wall', 'Wall', 'Wall', 'Wall', 'Wall', 'Wall', 'Wall']
]


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
from EduardMCGS.evaluation import Evaluation
from rl_agents.agents.common.factory import load_agent, load_environment
from Environments.DemoMiniGrid import DemoMiniGrid
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
    env.unwrapped = env.env.unwrapped
    env.observation_space = env.env.observation_space
    env.reward_range = env.env.reward_range
    env.metadata = env.env.metadata
    env.spec = gym.make('MiniGrid-DoorKey-8x8-v0').spec
    env.seed = env.env.seed
    env.render = env.env.render
    env.close = env.env.close

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
    if options["--train"]:
        images_per_episode = evaluation.train()
    elif options["--test"]:
        evaluation.test()
    else:
        evaluation.close()
    return os.path.relpath(evaluation.monitor.directory), images_per_episode, agent, env

env = DemoMiniGrid(default_ascii, seed=42)

agent_config = {
    "__class__": "<class 'rl_agents.agents.tree_search.graph_based.GraphBasedPlannerAgent'>",
    "gamma": 0.99,
}
options = {
    "--seed": 42,
    "--no-display": True,
    "--episodes": 1,
    "--train": True,
    "--test": True,
}

_, images_per_episode, agent, env = evaluate(env, agent_config, options)
for images in images_per_episode:
    plot_images(len(images), images, 0, True)