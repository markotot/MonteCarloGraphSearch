import os

from rl_agents.trainer import logger
from EdouardMCGS.evaluation import Evaluation
from rl_agents.agents.common.factory import load_agent
from Environments.MinigridLoggerMetrics import MinigridMetrics

def run_experiment(env, agent_config, options):
    #  to make it work with our MiniGridEnv
    # env.unwrapped = env.env.unwrapped
    # env.observation_space = env.env.observation_space
    # env.reward_range = env.env.reward_range
    # env.metadata = env.env.metadata
    # env.spec = env.env.spec
    # env.seed = env.env.seed
    # env.render = env.env.render
    # env.close = env.env.close
    # #  to make it work with our MiniGridEnv

    logger.configure()
    agent = load_agent(agent_config, env)
    options["--seed"] = int(options["--seed"]) if options["--seed"] is not None else None

    evaluation = Evaluation(
        env,
        agent,
        run_directory=None,
        num_episodes=int(options["--episodes"]),
        sim_seed=options["--seed"],
        recover=False,
        display_env=not options["--no-display"],
        display_agent=not options["--no-display"],
        display_rewards=not options["--no-display"],
    )

    images = evaluation.train()
    MinigridMetrics.save_metrics(agent, options["--seed"])
    return os.path.relpath(evaluation.monitor.directory), images, agent, env
