import numpy as np
import torch
from Gym_Environments.AbstractGymEnv import MyDoorKeyEnv

class ForwardModelEnv(MyDoorKeyEnv):

    correct_predictions = [0] * 7
    incorrect_predictions = [0] * 7

    data = []

    def __init__(self, model, device='cuda', size=8, collect=True, action_failure_prob=0, seed=42):
        super().__init__(size, action_failure_prob, seed)
        self.device = device
        self.model = model
        self.collect = collect

    def step(self, action):

        self.action = action  # Save the original action

        previous_observation = self.observation()
        surroundings = self.get_local_surrounding_processed(sight=3)

        self.state, self.reward, self.done, self.info = super().step(action)  # Do the step
        observation = self.observation()

        if self.collect:
            self.collect_data(previous_observation, observation, action, surroundings)
        else:
            self.model_step(previous_observation, action, surroundings, observation)

        return observation, self.reward, self.done, self.info

    def model_step(self, previous_observation, action, surroundings, label_observation):

        x = [
            previous_observation[0],
            previous_observation[1],
            previous_observation[2],
            int(previous_observation[3] is not None),
            int(previous_observation[4] is True),
            int(previous_observation[5] is True),
            action,
        ]

        x = np.concatenate((np.array(x), surroundings))

        y = [
            label_observation[0],
            label_observation[1],
            label_observation[2],
            int(label_observation[3] is not None),
            int(label_observation[4] is True),
            int(label_observation[5] is True),
        ]

        x = torch.tensor(x).type(torch.FloatTensor).to(self.device)
        output = self.model(x)
        model_observation = output.round().cpu().detach().numpy()

        if np.array_equal(model_observation, y):
            ForwardModelEnv.correct_predictions[action] += 1
        else:
            ForwardModelEnv.incorrect_predictions[action] += 1

        return model_observation


    def model_stats(self):
        print(f"Correct predictions: {ForwardModelEnv.correct_predictions}")
        print(f"Incorrect predictions: {ForwardModelEnv.incorrect_predictions}")

    def collect_data(self, state, next_state, action, surroundings):

        x = [
            state[0],
            state[1],
            state[2],
            int(state[3] is not None),
            int(state[4] is True),
            int(state[5] is True),
            action,
        ]

        y = [
            next_state[0],
            next_state[1],
            next_state[2],
            int(next_state[3] is not None),
            int(next_state[4] is True),
            int(next_state[5] is True),
            action,
        ]

        surroundings = surroundings.flatten()
        surroundings_string = ""
        for element in surroundings:
            surroundings_string += ", " + str(element)
        ForwardModelEnv.data.append((x, y, [action], surroundings_string))


    def save_data(self, file_path):
        f = open(file_path, "w")
        for data_point in ForwardModelEnv.data:
            features = data_point[0] + data_point[1] + data_point[2]
            surroundings = data_point[3]
            processed_data_point = str(features)[1:-1] + surroundings
            f.write(processed_data_point + "\n")
        f.close()
        ForwardModelEnv.data = []
