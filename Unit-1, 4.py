import numpy as np

class DroneGridMDP:
    def __init__(self, grid_size=5, warehouse=(0,0), deliveries=None, gamma=0.9):
        self.grid_size = grid_size
        self.warehouse = warehouse
        self.deliveries = deliveries if deliveries else [(4,4), (2,3)]
        self.gamma = gamma
        self.states = [(r,c) for r in range(grid_size) for c in range(grid_size)]
        self.actions = ["up","down","left","right"]

    def reward(self, state):
        if state in self.deliveries:
            return 10
        elif state == self.warehouse:
            return 0
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

def policy_improvement(env, V, policy):
    stable = True
    for s in env.states:
        old_action = max(policy[s], key=policy[s].get)
        # greedy action
        q_values = {}
        for a in env.actions:
            s_next = env.transition(s,a)
            r = env.reward(s_next)
            q_values[a] = r + env.gamma * V[s_next]
        best_action = max(q_values, key=q_values.get)
        # update policy
        for a in env.actions:
            policy[s][a] = 1.0 if a == best_action else 0.0
        if best_action != old_action:
            stable = False
    return policy, stable

def policy_iteration(env):
    # initialize uniform random policy
    policy = {s: {a: 1/len(env.actions) for a in env.actions} for s in env.states}
    while True:
        V = policy_evaluation(env, policy)
        policy, stable = policy_improvement(env, V, policy)
        if stable:
            return policy, V

# --- Example Usage ---
env = DroneGridMDP()
optimal_policy, optimal_values = policy_iteration(env)

print("Optimal Value Function:")
for r in range(env.grid_size):
    row_vals = [round(optimal_values[(r,c)],2) for c in range(env.grid_size)]
    print(row_vals)

print("\nOptimal Policy (greedy actions):")
for r in range(env.grid_size):
    row_actions = [max(optimal_policy[(r,c)], key=optimal_policy[(r,c)].get) for c in range(env.grid_size)]
    print(row_actions)
