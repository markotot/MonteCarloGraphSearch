

class AbstractNovelty:
    def __init__(self, config, agent):
        self.agent = agent
        self.env = agent.env

        self.total_data_points = 0
        self.novelty_function_name = config['novelty_function_name']
        self.novelty_percentage = config['novelty_percentage']

    def calculate_novelty(self, observation):
        return getattr(self, self.novelty_function_name)(observation=observation)

    def update_posterior(self, observation, step):
        raise NotImplementedError

    def get_metrics(self):
        raise NotImplementedError