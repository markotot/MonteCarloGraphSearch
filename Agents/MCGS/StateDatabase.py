import numpy as np
from Utils.Logger import Logger


class StateDatabase:

    def __init__(self, config):

        self.config = config

        self.x_pos = [0] * 16
        self.y_pos = [0] * 16
        self.rotation = [0] * 4
        self.agent_carry = {None: 0, 'key': 0}
        self.door_open = {True: 0, False: 0}

        self.total_data_points = 0
        self.novelty_function_name = config['novelty_function_name']
        self.novelty_percentage = config['novelty_percentage']

        self.subgoals = {
            'key_subgoal': -1,
            'door_subgoal': -1,
            'goal_found': -1,
        }

    def calculate_novelty(self, observation):
        return getattr(self, self.novelty_function_name)(observation=observation)

    def simple_novelty_function(self, *args, **kargs):
        observation = kargs['observation']
        novelty = [0] * 5
        if self.x_pos[observation[0]] < self.calculate_novelty_threshold() / len(self.x_pos):
            novelty[0] = 1
        if self.y_pos[observation[1]] < self.calculate_novelty_threshold() / len(self.y_pos):
            novelty[1] = 1
        if self.rotation[observation[2]] < self.calculate_novelty_threshold() / len(self.rotation):
            novelty[2] = 1
        if self.agent_carry[observation[3]] < self.calculate_novelty_threshold() / len(self.agent_carry):
            novelty[3] = 1
        if self.door_open[observation[4]] < self.calculate_novelty_threshold() / len(self.door_open):
            novelty[4] = 1
        return np.sum(novelty)

    def calculate_novelty_threshold(self):
        return self.total_data_points * self.novelty_percentage

    def update_posterior(self, observation):

        self.x_pos[observation[0]] += 1
        self.y_pos[observation[1]] += 1
        self.rotation[observation[2]] += 1
        self.agent_carry[observation[3]] += 1
        self.door_open[observation[4]] += 1

        self.total_data_points += 1

        if observation[3] == 'key':
            self.key_picked_up()
        if observation[4] is True:
            self.door_opened()

    def key_picked_up(self):
        if self.subgoals['key_subgoal'] == -1:
            self.subgoals['key_subgoal'] = self.total_data_points
            Logger.log_data(f"Key subgoal found (Total nodes: {self.total_data_points})")

    def door_opened(self):
        if self.subgoals['door_subgoal'] == -1:
            self.subgoals['door_subgoal'] = self.total_data_points
            Logger.log_data(f"Door subgoal found (Total nodes: {self.total_data_points})")

    def goal_found(self):
        if self.subgoals['goal_found'] == -1:
            self.subgoals['goal_found'] = self.total_data_points
            Logger.log_data(f"Goal found (Total nodes: {self.total_data_points})")

