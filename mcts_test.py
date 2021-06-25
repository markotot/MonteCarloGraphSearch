import os
from gym_minigrid.wrappers import *

from rl_agents.trainer import logger
from EdouardMCGS.evaluation import Evaluation
from rl_agents.agents.common.factory import load_agent, load_environment

from Environments.MiniGridEnv import MiniGridEnv
from Utils.Logger import plot_images

BENCHMARK_FILE = "benchmark_summary"
LOGGING_CONFIG = "configs/logging.json"
VERBOSE_CONFIG = "configs/verbose.json"

def evaluate(env, agent_config, options):
    """
        Evaluate an agent interacting with an environment.
    :param environment_config: the path of the environment configuration file
    :param agent_config: the path of the agent configuration file
    :param options: the evaluation options
    """
    env.unwrapped = env.env.unwrapped
    env.observation_space = env.env.observation_space
    env.reward_range = env.env.reward_range
    env.metadata = env.env.metadata
    env.spec = env.env.spec
    env.seed = env.env.seed
    env.render = env.env.render
    env.close = env.env.close

    logger.configure()
    # if options['--verbose']:
    #     logger.configure(VERBOSE_CONFIG)
    agent = load_agent(agent_config, env)
    run_directory = None
    # if options['--name-from-config']:
    #     run_directory = "{}_{}_{}".format(Path(agent_config).with_suffix('').name,
    #                               datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
    #                               os.getpid())
    options["--seed"] = (
        int(options["--seed"]) if options["--seed"] is not None else None
    )
    evaluation = Evaluation(
        env,
        agent,
        run_directory=run_directory,
        num_episodes=int(options["--episodes"]),
        sim_seed=options["--seed"],
        recover=False,  # options['--recover'] or options['--recover-from'],
        display_env=not options["--no-display"],
        display_agent=not options["--no-display"],
        display_rewards=not options["--no-display"],
    )
    if options["--train"]:
        images_per_episode = evaluation.train()
    elif options["--test"]:
        evaluation.test()
    else:
        evaluation.close()
    return os.path.relpath(evaluation.monitor.directory), images_per_episode, agent, env



env = MiniGridEnv('MiniGrid-DoorKey-5x5-v0')

agent_config = {
    "__class__": "<class 'rl_agents.agents.tree_search.mcts.MCTSAgent'>",
    "gamma": 0.99,
    "budget" : 500
}
options = {
    "--seed": 42,
    "--no-display": True,
    "--episodes": 1,
    "--train": True,
    "--test": True,
}


_, images_per_episode, agent, env = evaluate(env, agent_config, options)

for images in images_per_episode:
    plot_images(0, images, 0, True)
