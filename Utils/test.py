default_ascii = [
    ['Wall', 'Wall', 'Wall', 'Wall', 'Wall', 'Wall', 'Wall', 'Wall'],
    ['Wall', 'Player', ' ', 'Wall', ' ', ' ', ' ', 'Wall'],
    ['Wall', 'Key', ' ', 'Wall', ' ', ' ', ' ', 'Wall'],
    ['Wall', 'Wall', 'Door', 'Wall', ' ', ' ', ' ', 'Wall'],
    ['Wall', ' ', ' ', ' ', ' ', ' ', ' ', 'Wall'],
    ['Wall', ' ', ' ', ' ', ' ', ' ', ' ', 'Wall'],
    ['Wall', ' ', ' ', ' ', ' ', 'Goal', ' ', 'Wall'],
    ['Wall', 'Wall', 'Wall', 'Wall', 'Wall', 'Wall', 'Wall', 'Wall']
]

from Environments.MiniGridEnv import MiniGridEnv
from Utils.Logger import plot_images


env_name = 'MiniGrid-DoorKey-16x16-v0'
images = []
for seed in [7, 109, 3, 35, 121]:
    env = MiniGridEnv(env_name, seed=seed)
    images.append(env.render())
plot_images(0, images, 0, True)