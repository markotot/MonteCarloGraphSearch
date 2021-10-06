import pandas as pd

from Environments.AbstractEnv import AbstractEnv
from datetime import datetime


class Metrics:
    key_found = False
    key_found_fmc = -1
    key_found_steps = -1
    door_opened = False
    door_opened_fmc = -1
    door_opened_steps = -1
    goal_found = False
    goal_found_fmc = -1
    goal_found_steps = -1
    solved = False
    solved_fmc = -1
    solved_steps = -1

    @staticmethod
    def save_metrics(agent):
        metrics = dict(total_nodes=[len(agent.planner.nodes)],
                       forward_model_calls=[AbstractEnv.forward_model_calls],
                       key_found=[Metrics.key_found],
                       key_found_steps=[Metrics.key_found_steps],
                       key_found_fmc=[Metrics.key_found_fmc],
                       door_opened=[Metrics.door_opened],
                       door_opened_steps=[Metrics.door_opened_steps],
                       door_opened_fmc=[Metrics.door_opened_fmc],
                       goal_found=[Metrics.goal_found],
                       goal_found_steps=[Metrics.goal_found_steps],
                       goal_found_fmc=[Metrics.goal_found_fmc],
                       solved=[Metrics.solved],
                       solved_steps=[Metrics.solved_steps],
                       solved_fmc=[Metrics.solved_fmc]
                       )
        metrics_data_frame = pd.DataFrame(metrics)
        d4 = datetime.today().strftime("%b-%d-%Y-%H-%M-%S")
        metrics_data_frame.to_csv(f"Results/Experiments/sota-mcgs-stochastic/experiment_metrics_{d4}.csv")
