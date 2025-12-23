import numpy as np
import matplotlib.pyplot as plt

# Gridworld setup
grid_size = 4
goal_state = (3, 3)
gamma = 0.9  # discount factor

def step(state, action):
    i, j = state
    if action == 0:  # up
        i = max(i-1, 0)
    elif action == 1:  # down
        i = min(i+1, grid_size-1)
    elif action == 2:  # left
        j = max(j-1, 0)
    elif action == 3:  # right
        j = min(j+1, grid_size-1)
    next_state = (i, j)
    reward = 1 if next_state == goal_state else -0.01
    return next_state, reward

# Policy evaluation
def evaluate_policy(policy, iterations=100):
    V = np.zeros((grid_size, grid_size))
    for _ in range(iterations):
        new_V = np.zeros_like(V)
        for i in range(grid_size):
            for j in range(grid_size):
                state = (i, j)
                if state == goal_state:
                    continue
                action_probs = policy[i, j]
                value = 0
                for a, prob in enumerate(action_probs):
                    next_state, reward = step(state, a)
                    value += prob * (reward + gamma * V[next_state])
                new_V[i, j] = value
        V = new_V
    return V

# Define policies
random_policy = np.ones((grid_size, grid_size, 4)) / 4
greedy_policy = np.zeros((grid_size, grid_size, 4))
optimal_policy = np.zeros((grid_size, grid_size, 4))

# Greedy: always move right or down
for i in range(grid_size):
    for j in range(grid_size):
        if i < grid_size-1:
            greedy_policy[i, j, 1] = 0.5  # down
        if j < grid_size-1:
            greedy_policy[i, j, 3] = 0.5  # right

# Optimal: deterministic shortest path (down/right bias)
for i in range(grid_size):
    for j in range(grid_size):
        if i < grid_size-1 and j < grid_size-1:
            optimal_policy[i, j, 1] = 0.5
            optimal_policy[i, j, 3] = 0.5
        elif i < grid_size-1:
            optimal_policy[i, j, 1] = 1.0
        elif j < grid_size-1:
            optimal_policy[i, j, 3] = 1.0

# Evaluate policies
V_random = evaluate_policy(random_policy)
V_greedy = evaluate_policy(greedy_policy)
V_optimal = evaluate_policy(optimal_policy)

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15,5))
titles = ["Random Policy", "Greedy Policy", "Optimal Policy"]
values = [V_random, V_greedy, V_optimal]

for ax, val, title in zip(axes, values, titles):
    im = ax.imshow(val, cmap="coolwarm", origin="upper")
    ax.set_title(title)
    for i in range(grid_size):
        for j in range(grid_size):
            ax.text(j, i, f"{val[i,j]:.2f}", ha="center", va="center", color="black")
fig.colorbar(im, ax=axes.ravel().tolist())
plt.show()
