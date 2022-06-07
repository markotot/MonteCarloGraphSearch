from collections import deque
import gym
import gym_minigrid
import numpy as np
import sys
import unittest

import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
import ray.rllib.agents.ppo as ppo
from ray.rllib.utils.test_utils import check_learning_achieved, framework_iterator
from ray.rllib.utils.numpy import one_hot
from ray.tune import register_env
from Gym_Environments.AbstractGymEnv import MyDoorKeyEnv
config = ppo.DEFAULT_CONFIG.copy()


def env_creator(env_config):
    return MyDoorKeyEnv(size=8, action_failure_prob=0.0, seed=121)  # return an env instance

register_env("my_env", env_creator)
trainer = ppo.PPOTrainer(env="my_env")


print("My trainer end")