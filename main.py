import matplotlib.pyplot as plt
import numpy as np

from Agents.MCTS.MCTSAgent import MCTSAgent
from Agents.MCGS.MCGSAgent import MCGSAgent

from Environments.MiniGridEnv import MiniGridEnv
from Utils.Logger import Logger, plot_images

# TODO: BUGS -
#  1) should be fixed --- fricking OOP --- action for step is sometimes None during rollout, very rarely but can happen (16x16, episodes=5, num_rollouts=24, rollout_depth=200)
#  2) should be fixed - children_criteria is empty in self.select_child(node, criteria_"value") (16x16, episodes=5, num_rollouts=24, rollout_depth=200)
#  3) should be fixed - if something is marked as not reachable, it will never become reachable again (can be fixed, but takes a lot of computation)
#  4) should be fixed - optimize route after rollouts
#  5) should be fixed - !! important !! Action trajectory doesn't reflect the real state
#  6) should be fixed - circular parenting, infinite loop in backprop

# TODO: Improvements
#  1) done - get_optimal_action() based on the best node, not just the best child
#  2) implement softmax for select_from_frontier()
#  3) for atari we might not need deepcopy/dijkstra
#  4) try to make a summarization of the graph using loops/cliques
#  5) try a Value Function with exploration
#  6) compare with a state of the art MCTS

# TODO: restrictions
#  1) node can't have edge into itself (problem with empty frontier)
#  2) stochastic environment not supported
#  3) partial observable env not supported, states are fully representative
#  4) env isn't perfect for rollouts - more moves you make in the env, less the reward - meaning later rollouts give less reward

# TODO next:
#   gauss novelties?


if __name__ == "__main__":

    env = MiniGridEnv('MiniGrid-DoorKey-16x16-v0')
    env.get_action_list()

    Logger.setup(path="test")
    agent = MCGSAgent(env, episodes=10, num_rollouts=20, rollout_depth=50)

    print(agent.info())
    images = [env.render()]
    total_reward = 0

    plt.imshow(images[0])
    plt.show()
    for i in range(200):
        action = agent.plan(draw_graph=False)
        state, reward, done, info = agent.act(action)
        images.append(env.render())
        total_reward += reward

        print(f"{i}){' ' * (4 - len(str(i)))} "
              f"Action: {action}",
              f"Reward: {reward}")
        if done:
            break

    Logger.close()
    agent.graph.save_graph("graph")
    plot_images(images, total_reward)
