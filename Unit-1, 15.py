import numpy as np
import random
from collections import defaultdict

# --- Environment ---
class CustomerEnv:
    def __init__(self):
        # States: simplified customer risk levels (0=low, 1=medium, 2=high)
        self.states = [0, 1, 2]
        # Actions: retention strategies (0=none, 1=email, 2=discount)
        self.actions = [0, 1, 2]
        self.state = None

    def reset(self):
        self.state = random.choice(self.states)
        return self.state

    def step(self, action):
        # Reward probabilities based on state and action
        if self.state == 0:  # low risk
            reward = 1 if action in [0,1] else 0
        elif self.state == 1:  # medium risk
            reward = 1 if action in [1,2] else -1
        else:  # high risk
            reward = 1 if action == 2 else -1

        # Transition to next customer
        self.state = random.choice(self.states)
        return self.state, reward

# --- Policy (example: discount for high risk, email for medium, none for low) ---
def fixed_policy(state):
    if state == 0:
        return 0  # do nothing
    elif state == 1:
        return 1  # send email
    else:
        return 2  # offer discount

# --- Monte Carlo Policy Evaluation ---
def monte_carlo_evaluation(env, policy, episodes=1000, gamma=0.9):
    returns = defaultdict(list)
    V = defaultdict(float)

    for _ in range(episodes):
        state = env.reset()
        episode = []
        # simulate one episode of 10 steps
        for t in range(10):
            action = policy(state)
            next_state, reward = env.step(action)
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
        V[s] = np.mean(returns[s])
    return V

# --- Run Simulation ---
env = CustomerEnv()
V = monte_carlo_evaluation(env, fixed_policy, episodes=5000)

print("Estimated State-Value Function under Fixed Policy:")
for s in env.states:
    print(f"State {s} (risk level): V = {round(V[s],2)}")
