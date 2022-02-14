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

from ray import tune

class MyCallBack(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.deltas = []

    def on_postprocess_trajectory(
        self,
        *,
        worker,
        episode,
        agent_id,
        policy_id,
        policies,
        postprocessed_batch,
        original_batches,
        **kwargs
    ):
        pos = np.argmax(postprocessed_batch["obs"], -1)
        x, y = pos % 8, pos // 8
        self.deltas.extend((x ** 2 + y ** 2) ** 0.5)

    def on_sample_end(self, *, worker, samples, **kwargs):
        print("mean. distance from origin={}".format(np.mean(self.deltas)))
        self.deltas = []


class OneHotWrapper(gym.core.ObservationWrapper):
    def __init__(self, env, vector_index, framestack):
        super().__init__(env)
        self.framestack = framestack
        # 49=7x7 field of vision; 11=object types; 6=colors; 3=state types.
        # +4: Direction.
        self.single_frame_dim = 49 * (11 + 6 + 3) + 4
        self.init_x = None
        self.init_y = None
        self.x_positions = []
        self.y_positions = []
        self.x_y_delta_buffer = deque(maxlen=100)
        self.vector_index = vector_index
        self.frame_buffer = deque(maxlen=self.framestack)
        for _ in range(self.framestack):
            self.frame_buffer.append(np.zeros((self.single_frame_dim,)))

        self.observation_space = gym.spaces.Box(
            0.0, 1.0, shape=(self.single_frame_dim * self.framestack,), dtype=np.float32
        )

    def observation(self, obs):
        # Debug output: max-x/y positions to watch exploration progress.
        if self.step_count == 0:
            for _ in range(self.framestack):
                self.frame_buffer.append(np.zeros((self.single_frame_dim,)))
            if self.vector_index == 0:
                if self.x_positions:
                    max_diff = max(
                        np.sqrt(
                            (np.array(self.x_positions) - self.init_x) ** 2
                            + (np.array(self.y_positions) - self.init_y) ** 2
                        )
                    )
                    self.x_y_delta_buffer.append(max_diff)
                    # print(
                    #     "100-average dist travelled={}".format(
                    #         np.mean(self.x_y_delta_buffer)
                    #     )
                    # )
                    self.x_positions = []
                    self.y_positions = []
                self.init_x = self.agent_pos[0]
                self.init_y = self.agent_pos[1]

        # Are we carrying the key?
        # if self.carrying is not None:
        #    print("Carrying KEY!!")

        self.x_positions.append(self.agent_pos[0])
        self.y_positions.append(self.agent_pos[1])

        # One-hot the last dim into 11, 6, 3 one-hot vectors, then flatten.
        objects = one_hot(obs[:, :, 0], depth=11)
        colors = one_hot(obs[:, :, 1], depth=6)
        states = one_hot(obs[:, :, 2], depth=3)
        # Is the door we see open?
        # for x in range(7):
        #    for y in range(7):
        #        if objects[x, y, 4] == 1.0 and states[x, y, 0] == 1.0:
        #            print("Door OPEN!!")

        all_ = np.concatenate([objects, colors, states], -1)
        all_flat = np.reshape(all_, (-1,))
        direction = one_hot(np.array(self.agent_dir), depth=4).astype(np.float32)
        single_frame = np.concatenate([all_flat, direction])
        self.frame_buffer.append(single_frame)
        return np.concatenate(self.frame_buffer)


def env_maker(config):
    env = gym.make("MiniGrid-Empty-16x16-v0")
    return env


register_env("mini-grid", env_maker)

if __name__ == "__main__":

    env = gym.make("MiniGrid-Empty-16x16-v0")
    config = ppo.DEFAULT_CONFIG.copy()
    # config["exploration_config"] = {
    #     "type": "Curiosity",
    #     # For the feature NN, use a non-LSTM fcnet (same as the one
    #     # in the policy model).
    #     "eta": 0.1,
    #     "lr": 0.0003,  # 0.0003 or 0.0005 seem to work fine as well.
    #     "feature_dim": 64,
    #     # No actual feature net: map directly from observations to feature
    #     # vector (linearly).
    #     "feature_net_config": {
    #         "fcnet_hiddens": [],
    #         "fcnet_activation": "relu",
    #     },
    #     "sub_exploration": {
    #         "type": "StochasticSampling",
    #     },
    # }
    ray.init()
    trainer = ppo.PPOTrainer(
            env=env,
             )

    while True:
        print(trainer.train())
