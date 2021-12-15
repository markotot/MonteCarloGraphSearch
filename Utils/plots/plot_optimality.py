import matplotlib.pyplot as plt
import numpy as np

optimal = 40
seeds = range(50)
labels = seeds
goal_found = np.array([34,
                       14,
                       13,
                       9,
                       15,
                       6,
                       26,
                       14,
                       8,
                       19,
                       22,
                       44,
                       20,
                       28,
                       26,
                       6,
                       10,
                       9,
                       19,
                       28,
                       13,
                       10,
                       33,
                       13,
                       32,
                       11,
                       38,
                       15,
                       8,
                       47,
                       34,
                       4,
                       37,
                       14,
                       17,
                       29,
                       35,
                       27,
                       12,
                       15,
                       31,
                       39,
                       32,
                       12,
                       24,
                       39,
                       16,
                       17,
                       14,
                       10])
level_solved = np.array([87,
                         62,
                         57,
                         62,
                         69,
                         49,
                         72,
                         60,
                         45,
                         79,
                         75,
                         83,
                         96,
                         79,
                         67,
                         49,
                         49,
                         64,
                         62,
                         72,
                         58,
                         66,
                         82,
                         58,
                         83,
                         53,
                         77,
                         58,
                         57,
                         98,
                         73,
                         60,
                         77,
                         70,
                         61,
                         76,
                         83,
                         82,
                         51,
                         65,
                         79,
                         87,
                         71,
                         56,
                         70,
                         89,
                         65,
                         61,
                         64,
                         58])
width = 0.35  # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots()

ax.bar(labels, goal_found, width, label='Goal Found')
ax.bar(labels, level_solved - goal_found, width, bottom=goal_found, label='Solved')

ax.set_ylabel('Steps')
ax.set_xlabel('Agent Seed')
# ax.set_title('Steps required to find the goal and solve the level')
ax.legend()

mean_solved = np.mean(level_solved)
mean_goal_found = np.mean(goal_found)

# plt.axhline(y=mean_goal_found, linewidth=1, linestyle='dashed')
# plt.axhline(y=mean_solved, linewidth=1, color='orange')

plt.axhline(y=optimal, linewidth=2, linestyle='dashed', color='green', label='Optimal solution')
labels = ["Optimal Solution", "Goal Discovered", "Level Solved"]
handles, _ = ax.get_legend_handles_labels()
plt.legend(handles=handles, labels=labels, loc=0, bbox_to_anchor=(1.02, 1.1), ncol=3, fancybox=False, shadow=False)

print(handles)

plt.show()

path_length = level_solved - goal_found
print(np.mean(path_length))
print(np.std(path_length))
print(np.mean(goal_found))
print(np.std(goal_found))
print(np.mean(level_solved))
