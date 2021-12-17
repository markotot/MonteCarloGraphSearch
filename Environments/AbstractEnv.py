from gym import Env
import numpy as np

class AbstractEnv:

    #forward_model_calls = 0

    def __init__(self, env: Env, action_failure_prob=0, seed=42):

        self.action_failure_prob = action_failure_prob
        self.is_stochastic = self.action_failure_prob > 0
        self.seed = seed
        self.env = env

        self.random = np.random.RandomState(self.seed)

        self.action_space = self.env.action_space
        self.action = None
        self.state = None
        self.reward = None
        self.done = None
        self.info = None
        self.reset()

    def step(self, action, action_failure_prob=None, failed_action=None):
        #AbstractEnv.forward_model_calls += 1
        self.action = action    # Save the original action

        if self.is_stochastic:  # If the env is stochastic check if action should fail
            if action_failure_prob is None:  # If the chance is not given, determine it here
                action_failure_prob = self.random.random_sample()
                failed_action = self.random.choice(range(self.action_space.n))

            if action_failure_prob < self.action_failure_prob:  # If the action should fail, swap it here
                #action = failed_action
                action = 6

        self.state, self.reward, self.done, self.info = self.env.step(action)   # Do the step
        return self.state, self.reward, self.done, self.info


    def random_step(self, disabled_actions=[]):
        possible_actions = [x for x in range(self.action_space.n) if x not in disabled_actions]
        random_action = np.random.choice(possible_actions)
        return self.step(random_action)

    def reset(self):
        self.env.reset()
        self.state = None
        self.done = None
        self.reward = None
        self.info = None
        print("FMC reset")
        #AbstractEnv.forward_model_calls = 0

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
