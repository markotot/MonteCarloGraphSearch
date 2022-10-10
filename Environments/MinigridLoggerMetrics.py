import pandas as pd

from Environments.AbstractEnv import AbstractEnv
from datetime import datetime


class MinigridMetrics:

    step_count = 1

    key_found = False
    key_found_fmc = -1
    key_found_nodes = -1
    key_found_steps = -1
    door_opened = False
    door_opened_fmc = -1
    door_opened_nodes = -1
    door_opened_steps = -1
    goal_found = False
    goal_found_fmc = -1
    goal_found_nodes = -1
    goal_found_steps = -1
    solved = False
    solved_fmc = -1
    solved_nodes = -1
    solved_steps = -1

    @staticmethod
    def save_metrics(agent, seed):
        metrics = dict(
                    agent_type=[str(agent.PLANNER_TYPE).split('.')[-1].split('\'')[0]],
                    env_seed=[seed],
                    total_nodes=[len(agent.planner.nodes)],
                    forward_model_calls=[AbstractEnv.forward_model_calls],
                    key_found=[MinigridMetrics.key_found],
                    key_found_steps=[MinigridMetrics.key_found_steps],
                    key_found_nodes=[MinigridMetrics.key_found_nodes],
                    key_found_fmc=[MinigridMetrics.key_found_fmc],
                    door_opened=[MinigridMetrics.door_opened],
                    door_opened_steps=[MinigridMetrics.door_opened_steps],
                    door_opened_nodes=[MinigridMetrics.door_opened_nodes],
                    door_opened_fmc=[MinigridMetrics.door_opened_fmc],
                    goal_found=[MinigridMetrics.goal_found],
                    goal_found_steps=[MinigridMetrics.goal_found_steps],
                    goal_found_nodes=[MinigridMetrics.goal_found_nodes],
                    goal_found_fmc=[MinigridMetrics.goal_found_fmc],
                    solved=[MinigridMetrics.solved],
                    solved_steps=[MinigridMetrics.solved_steps],
                    solved_nodes=[MinigridMetrics.solved_nodes],
                    solved_fmc=[MinigridMetrics.solved_fmc],
                       )
        metrics_data_frame = pd.DataFrame(metrics)
        date_time = datetime.today().strftime("%b-%d-%Y-%H-%M-%S")
        metrics_data_frame.to_csv(f"Results/Experiments/sota-mcgs-stochastic/experiment_metrics_{date_time}.csv")
