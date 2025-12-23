import numpy as np
import random
import matplotlib.pyplot as plt

# --- Environment ---
class AdEnv:
    def __init__(self, ctrs):
        self.ctrs = ctrs  # true click-through rates
        self.n_ads = len(ctrs)

    def step(self, ad):
        """Simulate showing an ad and getting a click (1) or not (0)."""
        return 1 if random.random() < self.ctrs[ad] else 0

# --- Algorithms ---
def epsilon_greedy(env, episodes=1000, epsilon=0.1):
    rewards = np.zeros(env.n_ads)
    counts = np.zeros(env.n_ads)
    history = []

    for t in range(episodes):
        if random.random() < epsilon:
            ad = random.randint(0, env.n_ads-1)
        else:
            ad = np.argmax(rewards / (counts + 1e-5))
        r = env.step(ad)
        counts[ad] += 1
        rewards[ad] += r
        history.append(sum(rewards)/sum(counts))
    return history

def ucb(env, episodes=1000):
    rewards = np.zeros(env.n_ads)
    counts = np.zeros(env.n_ads)
    history = []

    for t in range(episodes):
        if t < env.n_ads:
            ad = t
        else:
            avg_reward = rewards / (counts + 1e-5)
            confidence = np.sqrt(2 * np.log(t+1) / (counts + 1e-5))
            ad = np.argmax(avg_reward + confidence)
        r = env.step(ad)
        counts[ad] += 1
        rewards[ad] += r
        history.append(sum(rewards)/sum(counts))
    return history

def thompson_sampling(env, episodes=1000):
    alpha = np.ones(env.n_ads)
    beta = np.ones(env.n_ads)
    history = []

    for t in range(episodes):
        samples = [np.random.beta(alpha[i], beta[i]) for i in range(env.n_ads)]
        ad = np.argmax(samples)
        r = env.step(ad)
        if r == 1:
            alpha[ad] += 1
        else:
            beta[ad] += 1
        history.append(sum(alpha-1)/t if t>0 else 0)
    return history

# --- Simulation ---
true_ctrs = [0.05, 0.12, 0.08, 0.20]  # simulated true CTRs for ads
env = AdEnv(true_ctrs)
episodes = 5000

hist_eps = epsilon_greedy(env, episodes, epsilon=0.1)
hist_ucb = ucb(env, episodes)
hist_ts = thompson_sampling(env, episodes)

# --- Plot Results ---
plt.figure(figsize=(10,6))
plt.plot(hist_eps, label="Epsilon-Greedy")
plt.plot(hist_ucb, label="UCB")
plt.plot(hist_ts, label="Thompson Sampling")
plt.xlabel("Episodes")
plt.ylabel("Average CTR")
plt.title("Bandit Algorithms for Ad Selection")
plt.legend()
plt.show()

print("Final CTRs:")
print(f"Epsilon-Greedy: {hist_eps[-1]:.3f}")
print(f"UCB: {hist_ucb[-1]:.3f}")
print(f"Thompson Sampling: {hist_ts[-1]:.3f}")
