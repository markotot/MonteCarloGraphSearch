import numpy as np
from Utils.Logger import Logger
from Novelty.AbstractNovelty import AbstractNovelty

class DoorKeyNovelty(AbstractNovelty):

    def __init__(self, config, agent):

        super().__init__(config, agent)

        self.x_pos = [0] * agent.env.width
        self.y_pos = [0] * agent.env.height
        self.rotation = [0] * 4
        self.agent_carry = {None: 0, 'key': 0}
        self.door_open = {True: 0, False: 0}

        self.total_data_points = 0
        self.novelty_function_name = config['novelty_function_name']
        self.novelty_percentage = config['novelty_percentage']

        # (Nodes, Steps, FMC)
        self.subgoals = {
            'key_subgoal': (-1, -1, -1),
            'door_subgoal': (-1, -1, -1),
            'goal_found': (-1, -1, -1),
        }

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

    def update_posterior(self, observation, step):

        self.x_pos[observation[0]] += 1
        self.y_pos[observation[1]] += 1
        self.rotation[observation[2]] += 1
        self.agent_carry[observation[3]] += 1
        self.door_open[observation[4]] += 1

        self.total_data_points += 1

        if observation[3] == 'key':
            self.key_picked_up(step)
        if observation[4] is True:
            self.door_opened(step)

    def key_picked_up(self, step):
        if self.subgoals['key_subgoal'] == (-1, -1, -1):
            self.subgoals['key_subgoal'] = (self.total_data_points, step, self.agent.forward_model_calls)
            Logger.log_data(f"Key subgoal found (Total nodes: {self.total_data_points})")

    def door_opened(self, step):
        if self.subgoals['door_subgoal'] == (-1, -1, -1):
            self.subgoals['door_subgoal'] = (self.total_data_points, step, self.agent.forward_model_calls)
            Logger.log_data(f"Door subgoal found (Total nodes: {self.total_data_points})")

    def goal_found(self, step):
        if self.subgoals['goal_found'] == (-1, -1, -1):
            self.subgoals['goal_found'] = (self.total_data_points, step, self.agent.forward_model_calls)
            Logger.log_data(f"Goal found (Total nodes: {self.total_data_points})")

    def get_metrics(self):
        return dict(
            key_found_nodes=self.subgoals['key_subgoal'][0],
            key_found_steps=self.subgoals['key_subgoal'][1],
            key_found_FMC=self.subgoals['key_subgoal'][2],
            door_found_nodes=self.subgoals['door_subgoal'][0],
            door_found_steps=self.subgoals['door_subgoal'][1],
            door_found_FMC=self.subgoals['door_subgoal'][2],
            goal_found_nodes=self.subgoals['goal_found'][0],
            goal_found_steps=self.subgoals['goal_found'][1],
            goal_found_FMC=self.subgoals['goal_found'][2],
        )
