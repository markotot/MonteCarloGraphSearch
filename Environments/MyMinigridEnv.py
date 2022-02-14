from copy import deepcopy
import numpy as np
import gym
from Environments.AbstractEnv import AbstractEnv
from gym_minigrid import minigrid
from enum import Enum
import cv2


class EnvType(Enum):
    Empty = 0
    DoorKey = 1


class MyMinigridEnv(AbstractEnv):
    def __init__(self, name, action_failure_prob=0, seed=42):
        env = gym.make(name)
        env.seed(seed)
        super().__init__(env=env, action_failure_prob=action_failure_prob, seed=seed)

        self.name = self.env.unwrapped.spec.id

        if "DoorKey" in name:
            self.env_type = EnvType.DoorKey
        elif "Empty" in name:
            self.env_type = EnvType.Empty
        else:
            assert False

    def get_action_list(self):
        print("0 - Turn left")
        print("1 - Turn right")
        print("2 - Move forward")
        print("3 - Pick up object")
        print("4 - Drop object")
        print("5 - Interact")
        print("6 - Done")

    def render(self):
        return self.env.render(mode='rgb_array', highlight=False)

    def image_observation(self, size):
        image = self.render()
        return cv2.resize(image, dsize=(size, size), interpolation=cv2.INTER_CUBIC)


    def get_agent_position(self):
        agent_pos_x = self.env.agent_pos[0]
        agent_pos_y = self.env.agent_pos[1]
        agent_dir = self.agent_rotation_mapper(self.env.agent_dir)
        return tuple([agent_pos_x, agent_pos_y, agent_dir])

    def agent_rotation_mapper(self, agent_dir):
        return {0: "right", 1: "down", 2: "left", 3: "up"}[agent_dir]

    def agent_action_mapper(self, action):
        return {0: "Turns left",
                1: "Turns right",
                2: "Moves forward",
                3: "Picks up object",
                4: "Drops object",
                5: "Interacts",
                6: "Done",
                }[action]

    def get_observation(self):
        observation = None
        if self.env_type == EnvType.Empty:
            observation = self.get_observation_empty()
        elif self.env_type == EnvType.DoorKey:
            observation = self.get_observation_door_key()
        else:
            assert False
        return observation

    def observation(self):
        return self.get_observation()

    def get_observation_door_key(self):

        agent_pos_x = self.env.agent_pos[0]
        agent_pos_y = self.env.agent_pos[1]
        agent_dir = self.env.agent_dir
        agent_carry = None if self.env.carrying is None else self.env.carrying.type

        doors_open = [tile.is_open for tile in self.env.grid.grid if tile is not None and tile.type == "door"]
        doors_locked = [tile.is_locked for tile in self.env.grid.grid if tile is not None and tile.type == "door"]
        grid = [('empty' if tile is None else tile.type) for tile in self.env.grid.grid]

        return tuple([agent_pos_x, agent_pos_y, agent_dir, agent_carry] + doors_open + doors_locked + grid)

    def get_observation_empty(self):
        agent_pos_x = self.env.agent_pos[0]
        agent_pos_y = self.env.agent_pos[1]
        agent_dir = self.env.agent_dir
        grid = [('empty' if tile is None else tile.type) for tile in self.env.grid.grid]

        return tuple([agent_pos_x, agent_pos_y, agent_dir] + grid)
