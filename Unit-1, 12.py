import numpy as np

class GridWorld:
    def __init__(self, grid_size=5, goal=(4,4), gamma=0.9):
        self.grid_size = grid_size
        self.goal = goal
        self.gamma = gamma
        self.states = [(r,c) for r in range(grid_size) for c in range(grid_size)]
        self.actions = ["up","down","left","right"]

    def reward(self, state):
        if state == self.goal:
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

def value_iteration(env, theta=1e-4):
    V = {s: 0 for s in env.states}
    policy = {s: random.choice(env.actions) for s in env.states}

    while True:
        delta = 0
        for s in env.states:
            v = V[s]
            q_values = []
            for a in env.actions:
                s_next = env.transition(s,a)
                r = env.reward(s_next)
                q_values.append(r + env.gamma * V[s_next])
            V[s] = max(q_values)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break

    # Extract optimal policy
    for s in env.states:
        q_values = {}
        for a in env.actions:
            s_next = env.transition(s,a)
            r = env.reward(s_next)
            q_values[a] = r + env.gamma * V[s_next]
        policy[s] = max(q_values, key=q_values.get)

    return policy, V

# --- Example Usage ---
import random
env = GridWorld()
optimal_policy, optimal_values = value_iteration(env)

print("Optimal Value Function:")
for r in range(env.grid_size):
    row_vals = [round(optimal_values[(r,c)],2) for c in range(env.grid_size)]
    print(row_vals)

print("\nOptimal Policy (best action at each cell):")
for r in range(env.grid_size):
    row_actions = [optimal_policy[(r,c)] for c in range(env.grid_size)]
    print(row_actions)
