import cv2
from gym_minigrid.minigrid import *

from Environments.MyMinigridEnv import MyMinigridEnv


class CustomDoorKeyEnv(MiniGridEnv):
    """
    Environment with a door and key, sparse reward
    """

    def __init__(self, ascii=None, size=8, seed=42):

        if ascii is not None:
            self.ascii = ascii
        else:
            self.ascii = door_key_level_one
        super().__init__(
            grid_size=size,
            max_steps=10*size*size,
            seed=seed,
        )

    def set_ascii(self, ascii):
        self.ascii = ascii

    def render(self):
        return super().render(mode='rgb_array', highlight=False)

    def image_observation(self, size):
        image = self.render()
        return cv2.resize(image, dsize=(size, size), interpolation=cv2.INTER_CUBIC)
    def _gen_grid(self, width, height):

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

        self.mission = "use the key to open the door and then get to the goal"


class CustomDoorKey(MyMinigridEnv):

    def __init__(self, ascii=None, action_failure_prob=0, size=16, seed=42):
        env = CustomDoorKeyEnv(ascii, size=size, seed=seed)

        super().__init__(name=f"MiniGrid-DoorKey-{size}x{size}-v0",
                         action_failure_prob=action_failure_prob,
                         seed=seed)

        self.name = f"CustomDoorKey-{size}x{size}-v0"
        self.env = env
        self.seed = seed



door_key_level_one = [
    ['W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W'],
    ['W', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'P', 'W'],
    ['W', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'W'],
    ['W', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'W'],
    ['W', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'W'],
    ['W', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'W'],
    ['W', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'W'],
    ['W', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'W'],
    ['W', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'W'],
    ['W', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'W'],
    ['W', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'W'],
    ['W', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'W', 'W', 'W', 'W'],
    ['W', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'D', ' ', ' ', 'W'],
    ['W', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'W', ' ', ' ', 'W'],
    ['W', 'K', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'W', ' ', 'G', 'W'],
    ['W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W'],
]
