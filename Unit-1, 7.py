import numpy as np
import matplotlib.pyplot as plt

class WarehouseMDP:
    def __init__(self, grid_size=5, deliveries=None, gamma=0.9):
        self.grid_size = grid_size
        self.deliveries = deliveries if deliveries else [(4,4), (2,3)]
        self.gamma = gamma
        self.states = [(r,c) for r in range(grid_size) for c in range(grid_size)]
        self.actions = ["up","down","left","right"]

    def reward(self, state):
        if state in self.deliveries:
            return 10
        else:
            return -1

    def transition(self, state, action):
        r, c = state
        if action == "up": r -= 1
        elif action == "down": r += 1
        elif action == "left": c -= 1
        elif action == "right": c += 1
        # boundary check
        if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
            return state
        return (r,c)

def policy_evaluation(env, policy, theta=1e-4):
    V = {s: 0 for s in env.states}
    while True:
        delta = 0
        for s in env.states:
            v = V[s]
            new_v = 0
            for a, prob in policy[s].items():
                s_next = env.transition(s,a)
                r = env.reward(s_next)
                new_v += prob * (r + env.gamma * V[s_next])
            V[s] = new_v
            delta = max(delta, abs(v - new_v))
        if delta < theta:
            break
    return V

# --- Example Usage ---
env = WarehouseMDP()

# Define two policies: random vs greedy toward deliveries
random_policy = {s: {a: 1/len(env.actions) for a in env.actions} for s in env.states}
greedy_policy = {}
for s in env.states:
    # greedy: always move right if possible, else down
    greedy_policy[s] = {a: 0 for a in env.actions}
    if s[1] < env.grid_size-1:
        greedy_policy[s]["right"] = 1.0
    else:
        greedy_policy[s]["down"] = 1.0

V_random = policy_evaluation(env, random_policy)
V_greedy = policy_evaluation(env, greedy_policy)

# --- Visualization ---
def plot_values(V, title):
    grid = np.zeros((env.grid_size, env.grid_size))
    for r in range(env.grid_size):
        for c in range(env.grid_size):
            grid[r,c] = V[(r,c)]
    plt.imshow(grid, cmap="coolwarm", interpolation="nearest")
    plt.colorbar()
    plt.title(title)
    plt.show()

plot_values(V_random, "Value Function under Random Policy")
plot_values(V_greedy, "Value Function under Greedy Policy")
