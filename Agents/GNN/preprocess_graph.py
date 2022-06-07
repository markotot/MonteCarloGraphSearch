import concurrent.futures
import math
from Agents.MCGS.Graph import Graph
import networkx as nx
import numpy as np
import tensorflow as tf
import random

from keras.models import Sequential
from keras.layers import Dense

from torch_geometric.utils.convert import from_networkx




def process_observations(node):

    observation = node.id
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


    if has_key and door_opened:
        classification = [0, 0, 1]
    elif has_key and not door_opened:
        classification = [0, 1, 0]
    elif not has_key:
        classification = [1, 0, 0]
    else:
        print("Error in the classification")


    return processed_observation, classification


seed = 42
config = {
    'amplitude_factor': 0.1,
    'noisy_min_value': 0.1,
}

graph = Graph(seed, config)
graph.load_graph("../../graph.gpickle")

pyg_graph = from_networkx(graph.graph)


adj_mat = nx.adjacency_matrix(graph.graph)
nodes_info = nx.get_node_attributes(graph.graph, 'info').values()



x = []
y = []
for node in nodes_info:
    obs, classification = process_observations(node)
    x.append(np.array(obs))
    y.append(np.array(classification))


features = np.array(x).astype('float32')
labels = np.array(y).astype('float32')

available_indices = range(len(nodes_info))
train_indices = random.sample(range(len(nodes_info)), 300)
available_indices = [x for x in available_indices if x not in train_indices]
val_indices = random.sample(available_indices, 500)
available_indices = [x for x in available_indices if x not in val_indices]
test_indices = available_indices

train_input = []
train_labels = []
validation_input = []
validation_labels = []
test_input = []
test_labels = []

for idx in train_indices:
    train_input.append(features[idx])
    train_labels.append(labels[idx])
    
for idx in val_indices:
    validation_input.append(features[idx])
    validation_labels.append(labels[idx])

for idx in test_indices:
    test_input.append(features[idx])
    test_labels.append(labels[idx])


print(features.shape)
print(labels.shape)

model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=len(x[0])))
model.add(Dense(units=len(y[0]), activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy')
model.fit(train_input, train_labels, batch_size=1, epochs=1)

y_pred = model.predict(validation_input)
y_pred = [1 if y >= 0.5 else 0 for y in y_pred]

diff = y_pred - validation_labels
diff = np.sum(np.abs(diff))
print(diff)



