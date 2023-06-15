import copy
import matplotlib.pyplot as plt
from Gym_Environments.AbstractGymEnv import MyDoorKeyEnv
import networkx as nx
import random
from datetime import datetime

env_seed = 35
env = MyDoorKeyEnv(size=16, action_failure_prob=0, seed=env_seed)
graph = nx.DiGraph()


plt.imshow(env.render())
plt.show()
plt.close()
starting_obs = env.get_observation()
starting_state = env.get_env_state()

graph.add_node(starting_obs, state=starting_state)

frontier = []
frontier.append(starting_obs)

i = 0
steps = 0
while len(frontier) > 0:
    obs = frontier[0]
    del frontier[0]

    env_state = nx.get_node_attributes(graph, "state")[obs]
    env.set_env_state(env_state)

    actions = list(range(env.action_space.n))
    random.shuffle(actions)
    for action in actions:
        env_copy = copy.deepcopy(env)
        observation, _, done, _ = env_copy.step(action)
        steps += 1
        if done:
            break
        state = env_copy.get_env_state()
        if not graph.has_node(observation):
            graph.add_node(observation, state=state)
            frontier.insert(0, observation)
    if done:
        break
    if steps % 1000 == 0:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(f"[{current_time}]\tSteps: {steps}\t Nodes: {len(graph.nodes)}\t Frontier: {len(frontier)}")

plt.imshow(env_copy.render())
plt.show()
plt.close()
print(f"Done")
print(f"Steps {steps}\t Nodes: {len(graph.nodes)} Frontier: {len(frontier)}")