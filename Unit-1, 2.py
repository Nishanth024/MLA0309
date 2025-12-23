import numpy as np

class WarehouseMDP:
    def __init__(self, grid_size=4, items=None, obstacles=None, goal=None, gamma=0.9):
        self.grid_size = grid_size
        self.items = items if items else [(1,1), (2,2)]
        self.obstacles = obstacles if obstacles else [(0,2), (3,1)]
        self.goal = goal if goal else (3,3)
        self.gamma = gamma
        self.states = [(r,c) for r in range(grid_size) for c in range(grid_size)]
        self.actions = ["up","down","left","right"]

    def reward(self, state):
        if state in self.items:
            return 2
        elif state == self.goal:
            return 5
        elif state in self.obstacles:
            return -2
        else:
            return 0

    def transition(self, state, action):
        r, c = state
        if action == "up": r -= 1
        elif action == "down": r += 1
        elif action == "left": c -= 1
        elif action == "right": c += 1

        # boundary check
        if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
            return state  # stay in place
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

# --- Example usage ---
env = WarehouseMDP()

# Define a uniform random policy (equal probability for all actions)
policy = {s: {a: 1/len(env.actions) for a in env.actions} for s in env.states}

V = policy_evaluation(env, policy)

print("Value Function:")
for r in range(env.grid_size):
    row_vals = [round(V[(r,c)],2) for c in range(env.grid_size)]
    print(row_vals)
