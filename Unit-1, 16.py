import numpy as np
import random
import matplotlib.pyplot as plt

# --- Environment ---
class LearningPlatformEnv:
    def __init__(self, true_probs):
        self.true_probs = true_probs  # true engagement probabilities
        self.n_contents = len(true_probs)

    def step(self, content):
        # Simulate user engagement (Bernoulli trial)
        return 1 if random.random() < self.true_probs[content] else 0

# --- Epsilon-Greedy Strategy ---
def epsilon_greedy(env, episodes=1000, epsilon=0.1):
    rewards = np.zeros(env.n_contents)
    counts = np.zeros(env.n_contents)
    history = []

    for t in range(episodes):
        if random.random() < epsilon:
            content = random.randint(0, env.n_contents-1)
        else:
            content = np.argmax(rewards / (counts + 1e-5))
        r = env.step(content)
        counts[content] += 1
        rewards[content] += r
        history.append(sum(rewards)/sum(counts))
    return history, rewards, counts

# --- Simulation ---
true_probs = [0.05, 0.12, 0.08, 0.20]  # simulated engagement rates for content
env = LearningPlatformEnv(true_probs)

runs = 10
episodes = 2000
all_histories = []

for run in range(runs):
    history, rewards, counts = epsilon_greedy(env, episodes, epsilon=0.1)
    all_histories.append(history)

# --- Analysis ---
avg_history = np.mean(all_histories, axis=0)

plt.figure(figsize=(10,6))
plt.plot(avg_history, label="Epsilon-Greedy (ε=0.1)")
plt.xlabel("Episodes")
plt.ylabel("Average Engagement Rate")
plt.title("Epsilon-Greedy Content Recommendation Performance")
plt.legend()
plt.show()

print("Final average engagement rate:", avg_history[-1])
