import random
from collections import defaultdict

class CallCenterEnv:
    def __init__(self, reps=3):
        self.reps = reps
        self.states = list(range(reps))  # number of free reps
        self.actions = list(range(reps)) # assign to rep index

    def step(self, state, action):
        # If rep is free, assign call successfully
        if action < state:
            reward = 1
            next_state = state - 1
        else:
            reward = -1  # bad assignment
            next_state = state
        return next_state, reward

    def reset(self):
        return self.reps

def monte_carlo_policy_evaluation(env, policy, episodes=1000, gamma=0.9):
    returns = defaultdict(list)
    V = defaultdict(float)

    for _ in range(episodes):
        state = env.reset()
        episode = []
        # simulate one episode
        while state > 0:
            action = policy(state)
            next_state, reward = env.step(state, action)
            episode.append((state, reward))
            state = next_state

        # compute returns
        G = 0
        for t in reversed(range(len(episode))):
            s, r = episode[t]
            G = gamma * G + r
            returns[s].append(G)

    # average returns
    for s in returns:
        V[s] = sum(returns[s]) / len(returns[s])
    return V

# --- Example Policies ---
def random_policy(state):
    return random.randint(0, state-1) if state > 0 else 0

def greedy_policy(state):
    return 0  # always assign to first available rep

# --- Simulation ---
env = CallCenterEnv(reps=3)
V_random = monte_carlo_policy_evaluation(env, random_policy, episodes=5000)
V_greedy = monte_carlo_policy_evaluation(env, greedy_policy, episodes=5000)

print("Value Function under Random Policy:", dict(V_random))
print("Value Function under Greedy Policy:", dict(V_greedy))
