import numpy as np
from Utils.Logger import  Logger
class StateDatabase:

    def __init__(self):

        self.x_pos = [0] * 16
        self.y_pos = [0] * 16
        self.rotation = [0] * 4
        self.agent_carry = {None: 0, 'key': 0}
        self.door_open = {True: 0, False: 0}

        self.total_data_points = 0
        self.novelty_percentage = 0.25

        self.key_subgoal_reached = False
        self.door_subgoal_reached = False
        self.goal_subgoal_reached = False

    def calculate_novelty(self, observation):

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
        if self.key_subgoal_reached is False:
            self.key_subgoal_reached = True
            Logger.log_data(f"Key subgoal found (Total nodes: {self.total_data_points})")

    def door_opened(self):
        if self.door_subgoal_reached is False:
            self.door_subgoal_reached = True
            Logger.log_data(f"Door subgoal found (Total nodes: {self.total_data_points})")

    def goal_found(self):
        if self.goal_subgoal_reached is False:
            self.goal_subgoal_reached = True
            Logger.log_data(f"Goal found (Total nodes: {self.total_data_points})")

