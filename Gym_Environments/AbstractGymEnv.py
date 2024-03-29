from gym_minigrid.envs.doorkey import *
import cv2
from enum import Enum
from Environments.MyMinigridEnv import EnvType

class MyDoorKeyEnv(DoorKeyEnv):
    """
    Environment with a door and key, sparse reward
    """
    forward_model_calls = 0
    def __init__(self, size=8, action_failure_prob=0, seed=42, ascii=None):

        self.ascii = ascii

        super().__init__(size)
        super().seed(seed)

        self.action_failure_prob = action_failure_prob
        self.is_stochastic = self.action_failure_prob > 0


        self.env_type = EnvType.DoorKey
        self.random = np.random.RandomState(seed)
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
        MyDoorKeyEnv.forward_model_calls += 1
        return observation, self.reward, self.done, self.info

    def _gen_grid(self, width, height):

        if self.ascii is None:
            super()._gen_grid(width, height)
        else:
            # Create an empty grid
            self.grid = Grid(width, height)

            # Generate the surrounding walls
            self.grid.wall_rect(0, 0, width, height)

            for j, ascii_row in enumerate(self.ascii):
                for i, object in enumerate(ascii_row):
                    if object == 'Goal' or object == 'G':
                        self.put_obj(Goal(), i, j)
                    elif object == 'Player' or object == 'P':
                        self.agent_pos = (i, j)
                        self.agent_dir = 0
                        self.grid.set(i, j, None)
                    elif object == 'Wall' or object == 'W':
                        self.grid.set(i, j, Wall())
                    elif object == 'Key' or object == 'K':
                        self.put_obj(Key('yellow'), i, j)
                    elif object == 'Door' or object == 'D':
                        self.put_obj(Door('yellow', is_locked=True), i, j)
                    elif object == ' ':
                        pass
                    else:
                        raise ValueError(f" {object} Received an unknown object")

            self.mission = "Use the key to open the door and then get to the goal"

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

    def get_local_surrounding(self, sight=1):

        surrounding = np.zeros((2 * sight + 1, 2 * sight + 1), dtype=int)

        player_column = self.agent_pos[0]
        player_row = self.agent_pos[1]
        grid = np.reshape(self.grid.grid, (self.height, self.width))

        row = 0
        column = 0
        for i in range(player_row - sight, player_row + sight + 1):
            for j in range(player_column - sight, player_column + sight + 1):

                if i < 0 or i >= self.height or j < 0 or j >= self.width:
                    surrounding[row][column] = -1
                else:
                    element = 'empty' if grid[i][j] is None else grid[i][j].type
                    if element == "empty":
                        surrounding[row][column] = 0
                    elif element == "wall":
                        surrounding[row][column] = 1
                    elif element == "key":
                        surrounding[row][column] = 2
                    elif element == "door":
                        surrounding[row][column] = 3
                    elif element == "goal":
                        surrounding[row][column] = 4
                column += 1

            column = 0
            row += 1

        return surrounding

    def process_grid(self):
        grid = np.reshape(self.grid.grid, (self.height, self.width))
        surrounding = np.zeros(grid.shape)
        for i in range(self.height):
            for j in range(self.width):
                element = 'empty' if grid[i][j] is None else grid[i][j].type
                if element == "empty":
                    surrounding[i][j] = 0
                elif element == "wall":
                    surrounding[i][j] = 1
                elif element == "key":
                    surrounding[i][j] = 2
                elif element == "door":
                    surrounding[i][j] = 3
                elif element == "goal":
                    surrounding[i][j] = 4

        return surrounding