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

graph = Graph()
graph.load_graph("test.gpickle")
env = MiniGridEnv('MiniGrid-DoorKey-16x16-v0')

graph.set_root_node(graph.get_node_info(env.get_observation()))

start = time.time()

x = nx.bfs_tree(graph.graph, graph.root_node.id).edges()
end = time.time()
print(end - start)


