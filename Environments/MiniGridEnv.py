from copy import deepcopy

import gym
import numpy as np
from gym_minigrid import minigrid
from Envs.EnvWrapper import AbstractEnv

class MiniGridEnv(AbstractEnv):

    def __init__(self, name):
        env = gym.make(name)
        super().__init__(env)

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
        info['solved'] = reward > 0 and self.env.steps_remaining > 0
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

        doors = [tile.is_open for tile in self.env.grid.grid if tile is not None and tile.type == "door"]
        grid = [('empty' if tile is None else tile.type) for tile in self.env.grid.grid]

        return tuple([agent_pos_x, agent_pos_y, agent_dir, agent_carry] + doors + grid)

    @staticmethod
    def state_to_obs(state):
        agent_pos_x, agent_pos_y, agent_dir, agent_carry, step_count, grid = state
        if not agent_carry is None: agent_carry = agent_carry.type
        doors = [tile.is_open for tile in grid if tile is not None and tile.type == "door"]
        grid = [('empty' if tile is None else tile.type) for tile in grid]
        return tuple([agent_pos_x, agent_pos_y, agent_dir, agent_carry] + doors + grid)

    def clone_state(self):
        agent_pos_x = self.env.agent_pos[0]
        agent_pos_y = self.env.agent_pos[1]
        agent_dir = self.env.agent_dir
        agent_carry = None if self.env.carrying is None else self.env.carrying

        grid = deepcopy(self.env.grid.grid)   # necessary not to loose the reference to open doors and picked keys

        return agent_pos_x, agent_pos_y, agent_dir, agent_carry, self.env.step_count, tuple(grid)

    def restore_state(self, state):
        agent_pos_x, agent_pos_y, agent_dir, agent_carry, step_count, grid = state

        assert type(agent_carry) != str, "the state was wrong"


        self.env.agent_pos[0] = agent_pos_x
        self.env.agent_pos[1] = agent_pos_y
        self.env.agent_dir = agent_dir

        self.env.carrying = None if self.env.carrying is None else agent_carry

        self.env.step_count = step_count

        self.env.grid.grid = list(grid)


if __name__ == '__main__':

    """
    Visual test to check whether the clone/restore functions work properly 
    """

    import matplotlib.pyplot as plt
    from Envs.MiniGridEnv import MiniGridEnv

    env = MiniGridEnv("MiniGrid-DoorKey-5x5-v0")
    env.reset()

    initial_state = env.clone_state()

    plt.imshow(env.render())
    plt.show()

    env.step(0)
    env.step(3)
    env.step(2)
    env.step(1)
    env.step(5)
    env.step(2)
    env.step(2)
    env.step(1)
    env.step(2)

    print(env.get_observation()[3])
    print(env.env.carrying)

    plt.imshow(env.render())
    plt.show()

    env.restore_state(initial_state)
    print(env.get_observation()[3])
    print(env.env.carrying)

    plt.imshow(env.render())
    plt.show()












