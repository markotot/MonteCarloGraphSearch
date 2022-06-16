import numpy as np
from torch.utils.data import DataLoader, random_split
from models import *
from datasets import *
import pandas as pd

obsolete_actions = [0] * 7
impactful_actions = [0] * 7

predictions = np.zeros((7, 4), dtype=int) # [7 actions][correct, incorrect, incorrect_on_correlated_features, incorrect_on_uncorrelated_features]

incorrectly_predicted_features = np.zeros((2, 6), dtype=int) # [correlated, uncorrelated][agent_x, agent_y, rotation, has_key, door open, door_locked]

confusion_matrix = {
    "true_positive": 0,
    "true_negative": 0,
    "false_positive": 0,
    "false_negative": 0,
}

def feature_to_text(feature):
    features_text = {
        0: "agent_x",
        1: "agent_y",
        2: "agent_dir",
        3: "agent_carry",
        4: "doors_open",
        5: "doors_locked",
    }
    return features_text[feature]


def action_to_text(action):
    actions_text = {
        0: "Turns left",
        1: "Turns right",
        2: "Moves forward",
        3: "Picks up object",
        4: "Drops object",
        5: "Interacts",
        6: "Done",
    }
    return actions_text[action]


def calculate_obsolete_actions(input, label, action):
    global obsolete_actions
    global impactful_actions

    if np.equal(input[0:6], label).all():
        obsolete_actions[action] += 1
    else:
        impactful_actions[action] += 1


def calculate_confusion_matrix(output, label, feature_idx):

    if output[feature_idx] == 1 and label[feature_idx] == 1:
        confusion_matrix['true_positive'] += 1
    if output[feature_idx] == 0 and label[feature_idx] == 0:
        confusion_matrix['true_negative'] += 1
    if output[feature_idx] == 1 and label[feature_idx] == 0:
        confusion_matrix['false_positive'] += 1
    if output[feature_idx] == 0 and label[feature_idx] == 1:
        confusion_matrix['false_negative'] += 1


def calculate_correlated_features_prediction(output, label, action):

    feature_list = []
    if action == 0 or action == 1:
        feature_list = [2]
    elif action == 2:
        feature_list = [0, 1]
    elif action == 3:
        feature_list = [3]
    elif action == 4:
        feature_list = [3]
    elif action == 5:
        feature_list = [4, 5]
    elif action == 6:
        feature_list = []

    correlated_output = []
    correlated_label = []
    for feature in feature_list:
        correlated_output.append(output[feature])
        correlated_label.append(label[feature])

    correct_features = np.equal(output, label)
    if correct_features.all():
        predictions[action][0] += 1  # correct
    else:
        predictions[action][1] += 1  # incorrect

        for i in range(len(correct_features)):
            if not correct_features[i]:
                if i in feature_list:
                    incorrectly_predicted_features[0, i] += 1  # incorrectly predicted correlated  feature
                else:
                    incorrectly_predicted_features[1, i] += 1  # incorrectly predicted uncorrelated feature

        if np.equal(correlated_label, correlated_output).all():
            predictions[action][3] += 1  # incorrect_on_uncorrelated_features
        else:
            predictions[action][2] += 1  # incorrect_on_correlated_features

def print_analysis():

    print(f"Action: {action_to_text(confusion_matrix_action)}\t Feature: {feature_to_text(feature_idx)}")
    print(f"Confusion matrix: {confusion_matrix}\n")

    # percentages = [0] * 7
    # for idx in range(len(predictions[:, 0])):
    #     percentages[idx] = predictions[:, 0] / (predictions[:, 0] + predictions[:, 1])
    #     percentages = np.round(percentages, 4)
    # print(f"Accuracy per actions percentages: {percentages}")

    df = pd.DataFrame(predictions.transpose(),
                      columns=["rot_l", "rot_r", "move", "pick", "drop", "interact", "nothing"],
                      index=['cor', 'incorrect', 'inc_cor', 'inc_unc'])
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)
    print(f"\nTotal correct: {np.sum(predictions[:, 0])}\tTotal incorrect: {np.sum(predictions[:, 1])}")
    print("\n")
    df = pd.DataFrame(np.array(incorrectly_predicted_features),
                      columns=["agent_x", "agent_y", "rot", "has_key", "door_open", "door_locked"],
                      index=['cor_feat', 'uncor_feat'])
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)

    print("\n")
    print(f"Impactful actions: {impactful_actions}\t\tTotal: {np.sum(impactful_actions)}")
    print(f"Obsolete actions: {obsolete_actions}\t\tTotal: {np.sum(obsolete_actions)}")


torch.manual_seed(0)
data_set = Gridworld_Local_SAS_Dataset(sight=3, val=True)
data_set_size = 10000
validation_set, test_set = random_split(data_set, [data_set_size, len(data_set) - data_set_size])
validation_loader = DataLoader(validation_set, batch_size=1, shuffle=True)
validation_loader = DataLoader(data_set, batch_size=1, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = data_set.x.shape[1]
output_size = data_set.y.shape[1]
hidden_size = 96
model = NN_Forward_Model(input_size, output_size, hidden_size).to(device)
model.load_state_dict(torch.load("model.ckpt"), strict=True)


confusion_matrix_action = 5
feature_idx = 5

for x, y in validation_loader:
    x = x.to(device)
    y = y.to(device)

    output = model(x)

    input_state = x.cpu().detach().numpy()[0]
    label_state = y.cpu().detach().numpy()[0]
    output_state = output.round().cpu().detach().numpy()[0]
    action = int(input_state[6])

    calculate_obsolete_actions(input_state, label_state, action)
    if action == confusion_matrix_action:
        calculate_confusion_matrix(output_state, label_state, feature_idx)

    calculate_correlated_features_prediction(output_state, label_state, action)

print_analysis()