import matplotlib.pyplot as plt
import numpy as np

optimal = 40
seeds = range(50)
labels = seeds
level_solved = np.array([47,
                       45,
                       60,
                       49,
                       55,
                       45,
                       53,
                       47,
                       49,
                       51,
                       49,
                       56,
                       45,
                       45,
                       52,
                       60,
                       59,
                       55,
                       45,
                       62,
                       53,
                       51,
                       47,
                       56,
                       52,
                       50,
                       45,
                       57,
                       60,
                       53,
                       47,
                       92,
                       55,
                       51,
                       61,
                       47,
                       43,
                       47,
                       59,
                       41,
                       52,
                       55,
                       53,
                       60,
                       55,
                       47,
                       51,
                       59,
                       47,
                       51])
goal_found = np.array([7,
                        5,
                        4,
                        7,
                        5,
                        6,
                        5,
                        5,
                        3,
                        6,
                        7,
                        4,
                        4,
                        3,
                        10,
                        4,
                        14,
                        5,
                        5,
                        18,
                        10,
                        5,
                        4,
                        4,
                        7,
                        6,
                        4,
                        8,
                        6,
                        8,
                        3,
                        51,
                        9,
                        12,
                        8,
                        4,
                        4,
                        8,
                        11,
                        4,
                        6,
                        4,
                        10,
                        8,
                        6,
                        4,
                        6,
                        5,
                        3,
                        5])

level_solved += 1
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
print(np.std(level_solved))
