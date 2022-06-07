import concurrent.futures
import math
from Agents.MCGS.Graph import Graph
import networkx as nx
import numpy as np
import tensorflow as tf
import random
from keras.models import Sequential
from keras.layers import Dense

def get_dataset(graph):
    edges_info = nx.get_edge_attributes(graph.graph, 'info').values()

    parent_nodes = []
    actions = []
    child_nodes = []

    for edge in edges_info:
        parent_nodes.append(edge.node_from)
        actions.append(edge.action)
        child_nodes.append(edge.node_to)

    return parent_nodes, actions, child_nodes

def process_observation_to_input(node):

    observation = node.id
    #x  + y  + rot + key + open + unlocked
    #16 + 16 + 4   + 1   + 1    + 1
    processed_observation = [0] * 39
    processed_observation[observation[0]] = 1
    processed_observation[16 + observation[1]] = 1
    processed_observation[32 + observation[2]] = 1

    if observation[3] is not None:
        processed_observation[36] = 1
    if observation[4] is True:
        processed_observation[37] = 1
    if observation[5] is True:
        processed_observation[38] = 1

    return processed_observation

def split_data(input, output, train_size=0.5, validation_size=0.3):
    dataset_size = len(input)
    available_indices = range(dataset_size)
    train_indices = random.sample(available_indices, int(dataset_size * train_size))
    available_indices = [x for x in available_indices if x not in train_indices]
    val_indices = random.sample(available_indices, int(dataset_size * validation_size))
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
        train_labels.append(output[idx])

    for idx in val_indices:
        validation_input.append(features[idx])
        validation_labels.append(output[idx])

    for idx in test_indices:
        test_input.append(features[idx])
        test_labels.append(output[idx])

    train_input = np.array(train_input)
    train_labels = np.array(train_labels)
    validation_input = np.array(validation_input)
    validation_labels = np.array(validation_labels)
    test_input = np.array(test_input)
    test_labels = np.array(test_labels)

    return train_input, train_labels, validation_input, validation_labels, test_input, test_labels



def process_input_to_observation(input):

    x_coord = -1
    y_coord = -1
    rotation = -1
    has_key = -1
    opened = -1
    unlocked = -1

    for idx in range(16):
        if input[idx] == 1:
            x_coord = idx
        if input[16 + idx] == 1:
            y_coord = idx

    for idx in range(4):
        if input[32 + idx]:
            rotation = idx;

    has_key = input[36]
    opened = input[37]
    unlocked = input[38]

    return np.array([x_coord, y_coord, rotation, has_key, opened, unlocked])


seed = 42
config = {
    'amplitude_factor': 0.1,
    'noisy_min_value': 0.1,
}

graph = Graph(seed, config)
graph.load_graph("../../graph.gpickle")

parent_nodes, actions, child_nodes = get_dataset(graph)
dataset_size = len(parent_nodes)

x = []
y = []

for i in range(dataset_size):
    parent_obs = process_observation_to_input(parent_nodes[i])
    child_obs = process_observation_to_input(child_nodes[i])

    action_obs = np.zeros(7)
    action_obs[actions[i]] = 1
    input = np.append(parent_obs, action_obs)
    x.append(np.array(input))
    y.append(np.array(child_obs))


features = np.array(x).astype('float32')
labels = np.array(y).astype('float32')

train_x, train_y, val_x, val_y, test_x, test_y = split_data(features, labels, train_size=0.5, validation_size=0.3)

model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=len(x[0])))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=len(y[0]), activation='sigmoid'))

model.compile(optimizer='adam', loss='mse')

model.fit(train_x, train_y, batch_size=32, epochs=10)


y_pred = model.predict(val_x)
y_pred = np.round(y_pred)

total_diff = 0
correct = 0
same_prediction = 0

for idx in range(len(val_x)):
    y_obs = process_input_to_observation(y_pred[idx])
    y_label = process_input_to_observation(val_x[idx])

    if np.sum(y_pred[idx][0:39] - val_x[idx][0:39]) == 0:
        same_prediction += 1
    diff = np.sum(np.abs(y_obs - y_label))
    if diff == 0:
        correct += 1
    total_diff += diff

print("total diff: ", total_diff)
print("same pred: ", (same_prediction / len(val_x)), "%")
print("correct pred: ", (correct / len(val_x)), "%")
print(total_diff / len(val_x))

