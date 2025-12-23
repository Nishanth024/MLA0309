import numpy as np
import matplotlib.pyplot as plt

# Simulated environment: each content has a true probability of being liked
true_probabilities = [0.05, 0.1, 0.2, 0.3, 0.4]  # 5 content options
n_contents = len(true_probabilities)
n_rounds = 10000

# --- UCB Algorithm ---
def run_ucb(n_rounds, n_contents, true_probabilities):
    rewards = np.zeros(n_contents)
    selections = np.zeros(n_contents)
    total_reward = 0
    history = []

    for t in range(1, n_rounds+1):
        if t <= n_contents:
            choice = t-1  # play each content once initially
        else:
            ucb_values = rewards/selections + np.sqrt(2*np.log(t)/selections)
            choice = np.argmax(ucb_values)

        # Simulate reward
        reward = np.random.rand() < true_probabilities[choice]
        rewards[choice] += reward
        selections[choice] += 1
        total_reward += reward
        history.append(total_reward)

    return history

# --- ε-Greedy Algorithm ---
def run_epsilon_greedy(n_rounds, n_contents, true_probabilities, epsilon=0.1):
    rewards = np.zeros(n_contents)
    selections = np.zeros(n_contents)
    total_reward = 0
    history = []

    for t in range(n_rounds):
        if np.random.rand() < epsilon:
            choice = np.random.randint(n_contents)  # explore
        else:
            choice = np.argmax(rewards / (selections + 1e-5))  # exploit

        reward = np.random.rand() < true_probabilities[choice]
        rewards[choice] += reward
        selections[choice] += 1
        total_reward += reward
        history.append(total_reward)

    return history

# --- Random Strategy ---
def run_random(n_rounds, n_contents, true_probabilities):
    total_reward = 0
    history = []
    for t in range(n_rounds):
        choice = np.random.randint(n_contents)
        reward = np.random.rand() < true_probabilities[choice]
        total_reward += reward
        history.append(total_reward)
    return history

# Run simulations
ucb_history = run_ucb(n_rounds, n_contents, true_probabilities)
eps_history = run_epsilon_greedy(n_rounds, n_contents, true_probabilities)
rand_history = run_random(n_rounds, n_contents, true_probabilities)

# Plot results
plt.figure(figsize=(10,6))
plt.plot(ucb_history, label="UCB")
plt.plot(eps_history, label="ε-Greedy (ε=0.1)")
plt.plot(rand_history, label="Random")
plt.xlabel("Rounds")
plt.ylabel("Cumulative Reward")
plt.title("Content Selection Strategies on Streaming Platform")
plt.legend()
plt.show()
