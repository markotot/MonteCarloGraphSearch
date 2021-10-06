import os

from rl_agents.trainer import logger
from EdouardMCGS.evaluation import Evaluation
from rl_agents.agents.common.factory import load_agent
from Environments.metrics import Metrics

def run_experiment(env, agent_config, options):
    #  to make it work with our MiniGridEnv
    env.unwrapped = env.env.unwrapped
    env.observation_space = env.env.observation_space
    env.reward_range = env.env.reward_range
    env.metadata = env.env.metadata
    env.spec = env.env.spec
    env.seed = env.env.seed
    env.render = env.env.render
    env.close = env.env.close
    #  to make it work with our MiniGridEnv

    logger.configure()
    agent = load_agent(agent_config, env)
    run_directory = None
    options["--seed"] = int(options["--seed"]) if options["--seed"] is not None else None

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

    images = evaluation.train()
    Metrics.save_metrics(agent)
    return os.path.relpath(evaluation.monitor.directory), images, agent, env
