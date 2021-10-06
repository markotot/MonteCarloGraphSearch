import numpy as np
from Utils.Logger import Logger
from Novelty.AbstractNovelty import AbstractNovelty

class EmptyNovelty(AbstractNovelty):

    def __init__(self, config, agent):

        super().__init__(config, agent)

        self.x_pos = [0] * agent.env.env.width
        self.y_pos = [0] * agent.env.env.height
        self.rotation = [0] * 4

        self.total_data_points = 0
        self.novelty_function_name = config['novelty_function_name']
        self.novelty_percentage = config['novelty_percentage']

        self.subgoals = {
            'goal_found': (-1, -1),
        }

    def simple_novelty_function(self, *args, **kargs):
        observation = kargs['observation']
        novelty = [0] * 3
        if self.x_pos[observation[0]] < self.calculate_novelty_threshold() / len(self.x_pos):
            novelty[0] = 1
        if self.y_pos[observation[1]] < self.calculate_novelty_threshold() / len(self.y_pos):
            novelty[1] = 1
        if self.rotation[observation[2]] < self.calculate_novelty_threshold() / len(self.rotation):
            novelty[2] = 1
        return np.sum(novelty)

    def calculate_novelty_threshold(self):
        return self.total_data_points * self.novelty_percentage

    def update_posterior(self, observation, step):

        self.x_pos[observation[0]] += 1
        self.y_pos[observation[1]] += 1
        self.rotation[observation[2]] += 1

        self.total_data_points += 1

    def goal_found(self, step):
        if self.subgoals['goal_found'] == (-1, -1, -1):
            self.subgoals['goal_found'] = (self.total_data_points, step, self.agent.forward_model_calls)
            Logger.log_data(f"Goal found (Total nodes: {self.total_data_points})")

    def get_metrics(self):
        return dict(
            goal_found_nodes=self.subgoals['goal_found'][0],
            goal_found_steps=self.subgoals['goal_found'][1],
            goal_found_FMC=self.subgoals['goal_found'][2],
        )

