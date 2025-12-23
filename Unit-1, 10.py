import numpy as np

# --- Environment ---
class MarketEnv:
    def __init__(self):
        self.states = ["bull", "bear", "neutral"]
        self.actions = ["stocks", "bonds", "cash"]
        self.state = None

    def reset(self):
        self.state = np.random.choice(self.states)
        return self.state

    def step(self, action):
        # Reward structure (simplified returns)
        if self.state == "bull":
            rewards = {"stocks": 5, "bonds": 2, "cash": 1}
        elif self.state == "bear":
            rewards = {"stocks": -4, "bonds": 3, "cash": 2}
        else:  # neutral
            rewards = {"stocks": 2, "bonds": 2, "cash": 2}

        reward = rewards[action]
        self.state = np.random.choice(self.states)  # next state
        return self.state, reward

# --- Policy Gradient Agent ---
class PolicyGradientAgent:
    def __init__(self, n_states, n_actions, lr=0.01, gamma=0.95):
        self.lr = lr
        self.gamma = gamma
        self.n_states = n_states
        self.n_actions = n_actions
        # policy parameters (state-action preferences)
        self.theta = np.random.rand(n_states, n_actions)

    def softmax(self, prefs):
        exp_prefs = np.exp(prefs - np.max(prefs))
        return exp_prefs / np.sum(exp_prefs)

    def get_action(self, state_idx):
        probs = self.softmax(self.theta[state_idx])
        action = np.random.choice(range(self.n_actions), p=probs)
        return action, probs

    def update(self, episode):
        G = 0
        for t in reversed(range(len(episode))):
            state_idx, action, reward, probs = episode[t]
            G = self.gamma * G + reward
            # Gradient update
            grad = -probs
            grad[action] += 1
            self.theta[state_idx] += self.lr * G * grad

# --- Simulation ---
env = MarketEnv()
agent = PolicyGradientAgent(n_states=3, n_actions=3)

episodes = 2000
returns = []

for ep in range(episodes):
    state = env.reset()
    state_idx = env.states.index(state)
    episode = []
    total_reward = 0

    for t in range(10):  # 10 steps per episode
        action_idx, probs = agent.get_action(state_idx)
        action = env.actions[action_idx]
        next_state, reward = env.step(action)
        next_state_idx = env.states.index(next_state)
        episode.append((state_idx, action_idx, reward, probs))
        total_reward += reward
        state_idx = next_state_idx

    agent.update(episode)
    returns.append(total_reward)

print("Average return after training:", np.mean(returns[-100:]))
