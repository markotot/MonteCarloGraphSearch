class AbstractAgent:

    def __init__(self, env):
        self.env = env

    def plan(self):
        pass

    def learn(self):
        pass

    def act(self, action):
        return self.env.step(action)

    def info(self):
        pass
