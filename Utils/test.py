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

class DemoDoorKeyEnv(MiniGridEnv):
    """
    Environment with a door and key, sparse reward
    """

    def __init__(self, ascii, size=8):
        self.ascii = ascii

        super().__init__(
            grid_size=size,
            max_steps=10*size*size
        )

    def _gen_grid(self, width, height):

        print(self.ascii)
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        for i, ascii_row in enumerate(self.ascii):
            for j, object in enumerate(ascii_row):
                if object == 'Goal':
                    self.put_obj(Goal(), i, j)
                elif object == 'Player':
                    self.agent_pos = (i, j)
                    self.agent_dir = self._rand_int(0, 4)
                    self.grid.set(i, j, None)
                elif object == 'Wall':
                    self.grid.set(i, j, Wall())
                elif object == 'Key':
                    self.put_obj(Key('yellow'), i, j)
                elif object == 'Door':
                    self.put_obj(Door('yellow', is_locked=True), i, j)
                elif object == ' ':
                    pass
                else:

                    raise ValueError(f" {object} Received an unknown object")

        self.mission = "use the key to open the door and then get to the goal"

env = DemoDoorKeyEnv(default_ascii)
