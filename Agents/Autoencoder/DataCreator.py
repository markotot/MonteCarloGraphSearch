from Environments.MyMinigridEnv import MyMinigridEnv
from gym_minigrid import minigrid
import matplotlib.pyplot as plt
import gym
import yaml


def load_agent_configuration(path):
    with open(path, 'r') as stream:
        return yaml.safe_load(stream)

def create_data(number_of_samples):
    images = []
    i = 0
    seed = 42
    env_name = "MiniGrid-DoorKey-16x16-v0"
    env = MyMinigridEnv(name=env_name, seed=seed)
    done = False
    while i < number_of_samples:

        if done:
            seed += 1
            env = MyMinigridEnv(name=env_name, seed=seed)

        env.random_step([6])
        image = env.image_observation(128)
        i += 1

        images.append(image)

    return images