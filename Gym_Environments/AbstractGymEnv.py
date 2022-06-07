from gym_minigrid.envs.doorkey import *
import cv2
from enum import Enum
from Environments.MyMinigridEnv import EnvType

class MyDoorKeyEnv(DoorKeyEnv):
    """
    Environment with a door and key, sparse reward
    """

    def __init__(self, size=8, action_failure_prob=0, seed=42):
        super().__init__(size)
        super().seed(seed)
        self.action_failure_prob = action_failure_prob
        self.is_stochastic = self.action_failure_prob > 0
        self.seed = seed
        # self.env = env
        self.env_type = EnvType.DoorKey
        self.random = np.random.RandomState(self.seed)
        self.name = "GymDoorkey"
        self.action = None
        self.state = None
        self.reward = None
        self.done = None
        self.info = None
        self.reset()

    def step(self, action):

        self.action = action  # Save the original action
        self.state, self.reward, self.done, self.info = super().step(action)  # Do the step

        observation = self.observation()
        return observation, self.reward, self.done, self.info

    def stochastic_step(self, action, action_failure_prob=None, failed_action=None):
        self.action = action  # Save the original action

        if self.is_stochastic:  # If the env is stochastic check if action should fail
            if action_failure_prob is None:  # If the chance is not given, determine it here
                action_failure_prob = self.random.random_sample()
                # action = self.random.choice(range(self.action_space.n))

            if action_failure_prob < self.action_failure_prob:  # If the action should fail, swap it here
                # action = failed_action
                action = 6  # No action

        self.state, self.reward, self.done, self.info = self.step(action)  # Do the step
        observation = self.observation()
        return observation, self.reward, self.done, self.info


    def reset(self):
        self.state = None
        self.done = None
        self.reward = None
        self.info = None
        super().reset()

        return self.observation()

    def render(self):
        return super().render(mode='rgb_array', highlight=False)

    def image_observation(self, size):
        image = self.render()
        return cv2.resize(image, dsize=(size, size), interpolation=cv2.INTER_CUBIC)

    def get_action_list(self):
        print("0 - Turn left")
        print("1 - Turn right")
        print("2 - Move forward")
        print("3 - Pick up object")
        print("4 - Drop object")
        print("5 - Interact")
        print("6 - Done")

    def get_agent_position(self):
        agent_pos_x = self.agent_pos[0]
        agent_pos_y = self.agent_pos[1]
        agent_dir = self.agent_rotation_mapper(self.agent_dir)
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

    def observation(self):
        return self.get_observation()

    def get_observation(self):
        agent_pos_x = self.agent_pos[0]
        agent_pos_y = self.agent_pos[1]
        agent_dir = self.agent_dir
        agent_carry = None if self.carrying is None else self.carrying.type

        doors_open = [tile.is_open for tile in self.grid.grid if tile is not None and tile.type == "door"]
        doors_locked = [tile.is_locked for tile in self.grid.grid if tile is not None and tile.type == "door"]
        grid = [('empty' if tile is None else tile.type) for tile in self.grid.grid]

        return tuple([agent_pos_x, agent_pos_y, agent_dir, agent_carry] + doors_open + doors_locked + grid)


