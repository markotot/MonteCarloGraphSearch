import pandas as pd
import matplotlib.pyplot as plt
from gym_minigrid.minigrid import *

from Environments.CustomDoorKeyEnv import CustomDoorKeyEnv

obsolete_actions = [0] * 7
impactful_actions = [0] * 7

# [7 actions] [correct, incorrect, incorrect_on_correlated_features, incorrect_on_uncorrelated_features]
predictions = np.zeros((7, 4), dtype=int)

# [correlated, uncorrelated] [agent_x, agent_y, rotation, has_key, door open, door_locked]
incorrectly_predicted_symbolic = np.zeros((2, 6), dtype=int)

# [obsolete, impactful] [correct_obsolete_prediction, correct_impactful_prediction, incorrect impactful_prediction]
symbolic_predictions = np.zeros((2, 3), dtype=int)
local_predictions = np.zeros((2, 3), dtype=int)
world_predictions = np.zeros((2, 3), dtype=int)

confusion_matrix = {
    "true_positive": 0,
    "true_negative": 0,
    "false_positive": 0,
    "false_negative": 0,
}

confusion_matrix_action = 5
feature_idx = 5

prediction_images = []
render_env = CustomDoorKeyEnv(ascii=None, size=16, seed=42)


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


def calculate_obsolete_actions(input, label, action, output_head=None):
    global obsolete_actions
    global impactful_actions

    if output_head is None or output_head == "symbolic":
        if np.equal(input[0:6], label[0:6]).all():
            obsolete_actions[action] += 1
        else:
            impactful_actions[action] += 1
    elif output_head == "local":
        if np.equal(input[7: 7 + 49], label).all():
            obsolete_actions[action] += 1
        else:
            impactful_actions[action] += 1
    elif output_head == "world":
        if np.equal(input[7 + 49: 7 + 49 + 256], label).all():
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
            if i < 6:
                if not correct_features[i]:
                    if i in feature_list:
                        incorrectly_predicted_symbolic[0, i] += 1  # incorrectly predicted correlated  feature
                    else:
                        incorrectly_predicted_symbolic[1, i] += 1  # incorrectly predicted uncorrelated feature

        if np.equal(correlated_label, correlated_output).all():
            predictions[action][3] += 1  # incorrect_on_uncorrelated_features
        else:
            predictions[action][2] += 1  # incorrect_on_correlated_features


def calculate_symbolic_prediction(input, output, label):
    obsolete = np.equal(input[0:6], label[0:6]).all()
    predicted_obsolete = np.equal(input[0:6], output[0:6]).all()
    correct = np.equal(output[0:6], label[0:6]).all()

    if obsolete:
        if correct:
            symbolic_predictions[0][0] += 1
        else:
            symbolic_predictions[0][2] += 1
    else:
        if correct:
            symbolic_predictions[1][1] += 1
        elif predicted_obsolete:
            symbolic_predictions[1][0] += 1
        else:
            symbolic_predictions[1][2] += 1


def print_symbolic(file):
    correct_symbolic = symbolic_predictions[0][0] + symbolic_predictions[1][1]
    incorrect_symbolic = symbolic_predictions[0][2] + symbolic_predictions[1][0] + symbolic_predictions[1][2]
    accuracy_symbolic = correct_symbolic / symbolic_predictions.sum()

    output_string = "-----------------------------------------------------------------\n"
    output_string += "Symbolic\n"
    output_string += f"Correct: {correct_symbolic}\tIncorrect: {incorrect_symbolic}\tAccuracy: {accuracy_symbolic}\n"
    df = pd.DataFrame(np.array(symbolic_predictions),
                      columns=["pred_obs", "pred_imp_cor", "pred_imp_inc"],
                      index=['obsolete', 'impactful'])
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        output_string += str(df)
    output_string += "\n"

    print(output_string)
    file.write(output_string)


def calculate_local_prediction(input, output, label):
    obsolete = np.equal(input[7: 7 + 49], label[6: 6 + 49]).all()
    predicted_obsolete = np.equal(input[7: 7 + 49], output[6: 6 + 49]).all()
    correct = np.equal(output[6: 6 + 49], label[6: 6 + 49]).all()

    # Only Local Model
    # obsolete = np.equal(input[7: 7 + 49], label).all()
    # predicted_obsolete = np.equal(input[7: 7 + 49], output).all()
    # correct = np.equal(output, label).all()

    if obsolete:
        if correct:
            local_predictions[0][0] += 1
        else:
            local_predictions[0][2] += 1
    else:
        if correct:
            local_predictions[1][1] += 1
        elif predicted_obsolete:
            local_predictions[1][0] += 1
        else:
            local_predictions[1][2] += 1


def print_local(file):
    correct_local = local_predictions[0][0] + local_predictions[1][1]
    incorrectly_local = local_predictions[0][2] + local_predictions[1][0] + local_predictions[1][2]
    accuracy_local = correct_local / local_predictions.sum()

    output_string = "-----------------------------------------------------------------\n"
    output_string += "Local\n"
    output_string += f"Correct: {correct_local}\tIncorrect: {incorrectly_local}\tAccuracy: {accuracy_local}\n"
    df = pd.DataFrame(np.array(local_predictions),
                      columns=["pred_obs", "pred_imp_cor", "pred_imp_inc"],
                      index=['obsolete', 'impactful'])
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        output_string += str(df)

    output_string += "\n"

    print(output_string)
    file.write(output_string)


def calculate_world_prediction(input, output, label, input_head, output_head):

    if output_head == "world":
        if input_head == "without_local":
            obsolete = np.equal(input[7:7 + 256], label).all()
            predicted_obsolete = np.equal(input[7:7 + 256], output).all()
            correct = np.equal(output, label).all()
        else:
            obsolete = np.equal(input[7 + 49:7 + 49 + 256], label).all()
            predicted_obsolete = np.equal(input[7 + 49:7 + 49 + 256], output).all()
            correct = np.equal(output, label).all()
    else:
        obsolete = np.equal(input[7 + 49:7 + 49 + 256], label[6 + 49: 6 + 49 + 256]).all()
        predicted_obsolete = np.equal(input[7 + 49:7 + 49 + 256], output[6 + 49: 6 + 49 + 256]).all()
        correct = np.equal(output[6 + 49:6 + 49 + 256], label[6 + 49: 6 + 49 + 256]).all()

    if obsolete:
        if correct:
            world_predictions[0][0] += 1
        else:
            if len(prediction_images) < 100:
                prediction_images.append(create_images(env=render_env, input=input, label=label, output=output, output_head=output_head))
            world_predictions[0][2] += 1
    else:
        if correct:
            world_predictions[1][1] += 1
        elif predicted_obsolete:
            world_predictions[1][0] += 1
        else:
            world_predictions[1][2] += 1


def print_world(file):
    correct_world = world_predictions[0][0] + world_predictions[1][1]
    incorrect_world = world_predictions[0][2] + world_predictions[1][0] + world_predictions[1][2]
    accuracy_world = correct_world / world_predictions.sum()

    output_string = "-----------------------------------------------------------------\n"
    output_string += "World\n"
    output_string += f"Correct: {correct_world}\tIncorrect: {incorrect_world}\tAccuracy: {accuracy_world}\n"
    df = pd.DataFrame(np.array(world_predictions),
                      columns=["pred_obs", "pred_imp_cor", "pred_imp_inc"],
                      index=['obsolete', 'impactful'])
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        output_string += str(df)

    print(output_string)
    file.write(output_string)


def calculate_head_predictions(input_state, output_state, label_state, output_head, input_head):
    if output_head == "symbolic":
        calculate_symbolic_prediction(input_state, output_state, label_state)
    elif output_head == "local":
        calculate_local_prediction(input_state, output_state, label_state)
    elif output_head == "world":
        calculate_world_prediction(input_state, output_state, label_state, input_head=input_head, output_head=output_head)
    else:
        calculate_symbolic_prediction(input_state, output_state, label_state)
        calculate_local_prediction(input_state, output_state, label_state)
        calculate_world_prediction(input_state, output_state, label_state, input_head=input_head, output_head=output_head)


def print_analysis(file):
    output_string = "-----------------------------------------------------------------\n"
    output_string += f"Action: {action_to_text(confusion_matrix_action)}\t Feature: {feature_to_text(feature_idx)}\n"
    output_string += f"Confusion matrix: {confusion_matrix}\n"

    output_string += "-----------------------------------------------------------------\n"
    df = pd.DataFrame(predictions.transpose(),
                      columns=["rot_l", "rot_r", "move", "pick", "drop", "interact", "nothing"],
                      index=['cor', 'incorrect', 'inc_cor', 'inc_unc'])
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        output_string += str(df)

    output_string += f"\n\nTotal correct: {np.sum(predictions[:, 0])}\tTotal incorrect: {np.sum(predictions[:, 1])}"
    output_string += f"\nAccuracy: {np.sum(predictions[:, 0]) / (np.sum(predictions[:, 0]) + np.sum(predictions[:, 1]))}"

    output_string += "\n-----------------------------------------------------------------\n"
    df = pd.DataFrame(np.array(incorrectly_predicted_symbolic),
                      columns=["agent_x", "agent_y", "rot", "has_key", "door_open", "door_locked"],
                      index=['cor_feat', 'uncor_feat'])
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        output_string += str(df)
    output_string += "\n-----------------------------------------------------------------\n"
    output_string += f"Impactful actions: {impactful_actions}\t\tTotal: {np.sum(impactful_actions)}\n"
    output_string += f"Obsolete actions: {obsolete_actions}\t\tTotal: {np.sum(obsolete_actions)}\n"
    print(output_string)
    file.write(output_string)


def create_images(env, input, label, output, output_head):

    if output_head == "world":
        world_index = 0
    else:
        world_index = 55

    symbolic = input[0:6]
    local = input[7:56]
    world = input[56:312]

    action = input[6]

    next_symbolic = label[0:6]
    next_local = label[7:56]
    next_world = label[world_index: world_index + 256]

    output_symbolic = output[0:6]
    output_local = output[6:55]
    output_world = output[world_index: world_index + 256]

    local_grid = local.reshape((7, 7))
    next_local_grid = next_local.reshape((7, 7))
    output_local_grid = output_local.reshape((7, 7))
    world_grid = world.reshape((16, 16))
    next_world_grid = next_world.reshape((16, 16))
    output_world_grid = output_world.reshape((16, 16))


    set_full_state(env, symbolic=symbolic, local=local_grid, world=world_grid)
    image_state = env.render()
    set_full_state(env, symbolic=next_symbolic, local=next_local_grid, world=next_world_grid)
    image_next_state = env.render()

    try:
        set_full_state(env, symbolic=output_symbolic, local=output_local_grid, world=output_world_grid)
        image_output_state = env.render()
    except Exception as e:
        print(e)
        image_output_state = image_next_state * 255

    return image_state, image_next_state, image_output_state, action

def set_full_state(env, symbolic, local, world):

    width = 16
    height = 16
    env.agent_pos = (int(symbolic[0]), int(symbolic[1]))
    env.agent_dir = int(symbolic[2])
    env.grid.set(int(symbolic[0]), int(symbolic[1]), None)

    env.grid = Grid(width, height)

    # Generate the surrounding walls
    env.grid.wall_rect(0, 0, width, height)

    for j, ascii_row in enumerate(world):
        for i, object in enumerate(ascii_row):
            if object == 4:
                env.put_obj(Goal(), i, j)
            elif object == 1:
                env.grid.set(i, j, Wall())
            elif object == 2:
                env.put_obj(Key('yellow'), i, j)
            elif object == 3:
                door = Door('yellow', is_locked=symbolic[5], is_open=symbolic[4])
                env.put_obj(door, i, j)
            elif object == 0:
                pass
            else:
                raise ValueError(f" {object} Received an unknown object")

    env.mission = "use the key to open the door and then get to the goal"

def plot_prediction_images(number=-1):
    for n in range(0, number):
        images = np.concatenate((prediction_images[n][0], prediction_images[n][1], prediction_images[n][2]), axis=1)
        plt.title(f"Action: {action_to_text(prediction_images[n][3])}")
        plt.imshow(images)
        plt.show()
