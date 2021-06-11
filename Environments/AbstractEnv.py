from gym import Env
import random

class AbstractEnv:

    def __init__(self, env: Env):

        self.env = env
        self.action_space = self.env.action_space
        self.action = None

        self.state = None
        self.reward = None
        self.done = None
        self.info = None

        self.reset()

    def step(self, action):

        self.action = action
        self.state, self.reward, self.done, self.info = self.env.step(self.action)
        return self.state, self.reward, self.done, self.info

    def random_step(self, disabled_actions=[]):
        possible_actions = [x for x in range(self.action_space.n) if x not in disabled_actions]
        
        self.action = random.choice(possible_actions)
        self.state, self.reward, self.done, self.info = self.env.step(self.action)
        return self.state, self.reward, self.done, self.info

    def reset(self):
        self.state = self.env.reset()
        self.done = None
        self.reward = None
        self.info = None

    def render(self):
        self.env.render()

    def get_state(self):
        return self.state

    def get_done(self):
        return self.done

    def get_reward(self):
        return self.reward

    def get_info(self):
        return self.info

    def get_action_list(self):
        raise NotImplementedError

    def get_observation(self):
        raise NotImplementedError

    def agent_rotation_mapper(self, agent_dir):
        raise NotImplementedError

    def agent_action_mapper(self, agent_dir):
        raise NotImplementedError
