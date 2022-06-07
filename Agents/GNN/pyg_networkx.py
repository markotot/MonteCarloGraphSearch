import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.utils.convert import from_networkx

def process_observation_ver_1(node):

    observation = node
    classification = [0] * 3
    has_key = False
    door_opened = False
    #x  + y  + rot + key + open + unlocked
    #16 + 16 + 4   + 1   + 1    + 1
    processed_observation = [0] * 39
    processed_observation[observation[0]] = 1
    processed_observation[16 + observation[1]] = 1
    processed_observation[32 + observation[2]] = 1

    if observation[3] is not None:
        processed_observation[36] = 1
        has_key = True
    if observation[4] is True:
        processed_observation[37] = 1
        door_opened = True
    if observation[5] is True:
        processed_observation[38] = 1
        pass


    # if has_key and door_opened:
    #     classification = [0, 0, 1]
    # elif has_key and not door_opened:
    #     classification = [0, 1, 0]
    # elif not has_key:
    #     classification = [1, 0, 0]
    # else:
    #     print("Error in the classification")

    return processed_observation

def process_observation_ver2(node):
    observation = node
    processed_observation = [observation[0],
                             observation[1],
                             observation[2],
                             int(observation[3] is not None),
                             int(observation[4] is True),
                             int(observation[5] is True)]
    return processed_observation

def action_to_one_hot(action):
    actions = [0] * 7
    actions[action] = 1
    return actions

from os import listdir
from os.path import isfile, join

all_files = [f for f in listdir("../../Data/Graphs/") if isfile(join("../../Data/Graphs/", f))]

parents = []
children = []
actions = []
for filename in all_files:
    print(filename)
    G = nx.readwrite.read_gpickle(f"../../Data/Graphs/{filename}")

    # remove info from Nodes and Edges so that it can be converted into pyG
    for (n, d) in G.nodes(data=True):
        del d["info"]
    for (n1, n2, d) in G.edges(data=True):

        processed_parent = process_observation_ver2(d['info'].node_from.id)
        # processed_parent = process_observation_ver_1(d['info'].node_from.id)
        processed_child = process_observation_ver2(d['info'].node_to.id)
        # processed_child = process_observation_ver_1(d['info'].node_to.id)
        parents.append(str(processed_parent)[1:-1])
        children.append(str(processed_child)[1:-1])

        action = d['info'].action
        # action = action_to_one_hot(d['info'].action)
        actions.append(str(action))
        #actions.append(str(action)[1:-1])
        d.clear()

    # adj_matrix_processed = []
    # adj_matrix = nx.adjacency_matrix(G)
    #
    # for idx, adj in enumerate(adj_matrix):
    #     splits = str(adj).split('\n')
    #     for x in splits:
    #         temp = x.split(',')
    #         if len(temp) < 2:
    #             continue
    #         temp = temp[1].split(')')[0]
    #         temp = str(idx) + ", " + temp
    #         adj_matrix_processed.append(temp)



    # node_attributes = []
    # for node in G.nodes:
    #     node_str = str(process_observation_ver2(node))[1:-1]
    #     node_attributes.append(node_str)
    #
    #

    save_path = "../../Data/Processed/" + filename.split(".")[0]

    # f = open(f"{save_path}_gridworld_A.txt", "w")
    # for adj in adj_matrix_processed:
    #     f.write(adj + "\n")
    # f.close()
    #
    # f = open(f"{save_path}_gridworld_node_attributes.txt", "w")
    # for node in node_attributes:
    #     f.write(node + "\n")
    # f.close()
print("Graphs processed. Saving to file")

f = open(f"../../Data/Datasets/dataset_16x16_SSA.txt", "w")
for idx in range(len(actions)):
    f.write(children[idx] + ", " + parents[idx] + ", " + actions[idx] + "\n")
f.close()
print("Done")