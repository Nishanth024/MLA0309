import numpy as np
import matplotlib.pyplot as plt

# --- Environment Setup ---
np.random.seed(42)
true_conversion_rates = [0.05, 0.12, 0.18, 0.25, 0.35]  # 5 campaigns
n_campaigns = len(true_conversion_rates)
n_rounds = 10000

# --- ε-Greedy ---
def run_epsilon_greedy(n_rounds, n_campaigns, true_rates, epsilon=0.1):
    rewards = np.zeros(n_campaigns)
    selections = np.zeros(n_campaigns)
    total_reward = 0
    history = []

    for t in range(n_rounds):
        if np.random.rand() < epsilon:
            choice = np.random.randint(n_campaigns)  # explore
        else:
            choice = np.argmax(rewards / (selections + 1e-5))  # exploit

        reward = np.random.rand() < true_rates[choice]
        rewards[choice] += reward
        selections[choice] += 1
        total_reward += reward
        history.append(total_reward)

    return history

# --- UCB ---
def run_ucb(n_rounds, n_campaigns, true_rates):
    rewards = np.zeros(n_campaigns)
    selections = np.zeros(n_campaigns)
    total_reward = 0
    history = []

    for t in range(1, n_rounds+1):
        if t <= n_campaigns:
            choice = t-1
        else:
            ucb_values = rewards/selections + np.sqrt(2*np.log(t)/selections)
            choice = np.argmax(ucb_values)

        reward = np.random.rand() < true_rates[choice]
        rewards[choice] += reward
        selections[choice] += 1
        total_reward += reward
        history.append(total_reward)

    return history

# --- Thompson Sampling ---
def run_thompson_sampling(n_rounds, n_campaigns, true_rates):
    successes = np.zeros(n_campaigns)
    failures = np.zeros(n_campaigns)
    total_reward = 0
    history = []

    for t in range(n_rounds):
        sampled_theta = [np.random.beta(successes[i]+1, failures[i]+1) for i in range(n_campaigns)]
        choice = np.argmax(sampled_theta)

        reward = np.random.rand() < true_rates[choice]
        if reward:
            successes[choice] += 1
        else:
            failures[choice] += 1

        total_reward += reward
        history.append(total_reward)

    return history

# --- Run Simulations ---
eps_history = run_epsilon_greedy(n_rounds, n_campaigns, true_conversion_rates, epsilon=0.1)
ucb_history = run_ucb(n_rounds, n_campaigns, true_conversion_rates)
ts_history = run_thompson_sampling(n_rounds, n_campaigns, true_conversion_rates)

# --- Plot Results ---
plt.figure(figsize=(10,6))
plt.plot(eps_history, label="ε-Greedy (ε=0.1)")
plt.plot(ucb_history, label="UCB")
plt.plot(ts_history, label="Thompson Sampling")
plt.xlabel("Rounds")
plt.ylabel("Cumulative Conversions")
plt.title("Marketing Campaign Optimization: Bandit Algorithms")
plt.legend()
plt.show()
