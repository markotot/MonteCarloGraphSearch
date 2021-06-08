from copy import deepcopy

import gym
import numpy as np
from gym_minigrid import minigrid
from Environments.AbstractEnv import AbstractEnv
from gym_minigrid.minigrid import *



class DemoDoorKeyEnv(MiniGridEnv):
    """
    Environment with a door and key, sparse reward
    """

    def __init__(self, ascii, size=8):
        self.ascii = ascii
        super().__init__(
            grid_size=size,
            max_steps=10*size*size
        )

    def _gen_grid(self, width, height):

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        for j, ascii_row in enumerate(self.ascii):
            for i, object in enumerate(ascii_row):
                if object == 'Goal':
                    self.put_obj(Goal(), i, j)
                elif object == 'Player':
                    self.agent_pos = (i, j)
                    self.agent_dir = self._rand_int(0, 4)
                    self.grid.set(i, j, None)
                elif object == 'Wall':
                    self.grid.set(i, j, Wall())
                elif object == 'Key':
                    self.put_obj(Key('yellow'), i, j)
                elif object == 'Door':
                    self.put_obj(Door('yellow', is_locked=True), i, j)
                elif object == ' ':
                    pass
                else:

                    raise ValueError(f" {object} Received an unknown object")

        self.mission = "use the key to open the door and then get to the goal"

class DemoMiniGrid(AbstractEnv):

    def __init__(self, ascii):
        env = DemoDoorKeyEnv(ascii)
        super().__init__(env)
        self.name = "DoorKeyDemo"
        self.env = env


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

    def step(self, action):
        (observation, reward, done, info) = self.env.step(action)
        return observation, reward, done, info

    def get_agent_position(self):
        agent_pos_x = self.env.agent_pos[0]
        agent_pos_y = self.env.agent_pos[1]
        agent_dir = self.agent_rotation_mapper(self.env.agent_dir)
        return tuple([agent_pos_x, agent_pos_y, agent_dir])

    def agent_rotation_mapper(self, agent_dir):
        return {0: "right", 1: "down", 2: "left", 3: "up"}[agent_dir]

    def agent_action_mapper(self, action):
        return {0: "Turn left",
                1: "Turn right",
                2: "Move forward",
                3: "Pick up object",
                4: "Drop object",
                5: "Interact",
                6: "Done",
                }[action]

    def get_observation(self):
        agent_pos_x = self.env.agent_pos[0]
        agent_pos_y = self.env.agent_pos[1]
        agent_dir = self.env.agent_dir
        agent_carry = None if self.env.carrying is None else self.env.carrying.type

        doors_open = [tile.is_open for tile in self.env.grid.grid if tile is not None and tile.type == "door"]
        doors_locked = [tile.is_locked for tile in self.env.grid.grid if tile is not None and tile.type == "door"]
        grid = [('empty' if tile is None else tile.type) for tile in self.env.grid.grid]

        return tuple([agent_pos_x, agent_pos_y, agent_dir, agent_carry] + doors_open + doors_locked + grid)




