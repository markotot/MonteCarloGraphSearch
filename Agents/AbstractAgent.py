import numpy as np

class AbstractAgent:

    def __init__(self, env, seed, verbose):
        self.env = env
        self.seed = seed
        self.verbose = verbose
        self.random = np.random.RandomState(self.seed)

    def plan(self):
        pass

    def learn(self):
        pass

    def act(self, action):
        return self.env.step(action)

    def info(self):
        pass
