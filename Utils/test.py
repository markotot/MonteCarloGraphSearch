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

import shutil
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
from Environments.DemoMiniGrid import DemoMiniGrid
from Environments.MiniGridEnv import MiniGridEnv
from Utils.Logger import plot_images


env_name = 'MiniGrid-DoorKey-16x16-v0'
images = []
for seed in [7, 109, 3, 35, 121]:
    env = MiniGridEnv(env_name, seed=seed)
    images.append(env.render())
plot_images(0, images, 0, True)