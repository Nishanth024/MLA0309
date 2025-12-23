import numpy as np
import random

# --- Environment ---
class PricingEnv:
    def __init__(self, prices, probs):
        self.prices = prices
        self.probs = probs  # acceptance probability for each price
        self.n_arms = len(prices)

    def step(self, arm):
        """Simulate customer response to chosen price arm"""
        if random.random() < self.probs[arm]:
            return self.prices[arm]  # revenue if accepted
        else:
            return 0  # no sale

# --- Strategies ---
def epsilon_greedy(env, episodes=1000, epsilon=0.1):
    rewards = np.zeros(env.n_arms)
    counts = np.zeros(env.n_arms)
    total_revenue = 0

    for t in range(episodes):
        if random.random() < epsilon:
            arm = random.randint(0, env.n_arms-1)
        else:
            arm = np.argmax(rewards / (counts + 1e-5))
        r = env.step(arm)
        counts[arm] += 1
        rewards[arm] += r
        total_revenue += r
    return total_revenue

def ucb(env, episodes=1000):
    rewards = np.zeros(env.n_arms)
    counts = np.zeros(env.n_arms)
    total_revenue = 0

    for t in range(episodes):
        if t < env.n_arms:
            arm = t
        else:
            avg_reward = rewards / (counts + 1e-5)
            confidence = np.sqrt(2 * np.log(t+1) / (counts + 1e-5))
            arm = np.argmax(avg_reward + confidence)
        r = env.step(arm)
        counts[arm] += 1
        rewards[arm] += r
        total_revenue += r
    return total_revenue

def thompson_sampling(env, episodes=1000):
    alpha = np.ones(env.n_arms)
    beta = np.ones(env.n_arms)
    total_revenue = 0

    for t in range(episodes):
        samples = [np.random.beta(alpha[i], beta[i]) for i in range(env.n_arms)]
        arm = np.argmax(samples)
        r = env.step(arm)
        if r > 0:
            alpha[arm] += 1
        else:
            beta[arm] += 1
        total_revenue += r
    return total_revenue

# --- Simulation ---
prices = [10, 20, 30, 40]
probs = [0.9, 0.6, 0.4, 0.2]  # acceptance probabilities

env = PricingEnv(prices, probs)

episodes = 5000
rev_eps = epsilon_greedy(env, episodes, epsilon=0.1)
rev_ucb = ucb(env, episodes)
rev_ts = thompson_sampling(env, episodes)

print("Total Revenue Comparison:")
print(f"Epsilon-Greedy: {rev_eps}")
print(f"UCB: {rev_ucb}")
print(f"Thompson Sampling: {rev_ts}")
