import numpy as np
import random

# --- Environment ---
class ManufacturingEnv:
    def __init__(self):
        # States: machine condition levels (0=poor, 1=average, 2=good)
        self.states = [0, 1, 2]
        # Actions: machine settings (0=low, 1=medium, 2=high)
        self.actions = [0, 1, 2]
        self.state = None

    def reset(self):
        self.state = random.choice(self.states)
        return self.state

    def step(self, action):
        # Reward based on how well action matches state
        if self.state == 0 and action == 0:
            reward = 2
        elif self.state == 1 and action == 1:
            reward = 5
        elif self.state == 2 and action == 2:
            reward = 10
        else:
            reward = -2  # poor quality if mismatch

        # Transition: next state randomly influenced by action
        self.state = random.choice(self.states)
        return self.state, reward

# --- Policy (ε-greedy) ---
class Policy:
    def __init__(self, n_states, n_actions, epsilon=0.1):
        self.epsilon = epsilon
        self.n_states = n_states
        self.n_actions = n_actions
        self.Q = np.zeros((n_states, n_actions))  # value function

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(range(self.n_actions))
        else:
            return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, gamma=0.9, alpha=0.1):
        best_next = np.max(self.Q[next_state])
        td_target = reward + gamma * best_next
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += alpha * td_error

# --- Simulation ---
env = ManufacturingEnv()
policy = Policy(n_states=3, n_actions=3)

episodes = 500
rewards = []

for ep in range(episodes):
    state = env.reset()
    total_reward = 0
    for t in range(20):  # steps per episode
        action = policy.select_action(state)
        next_state, reward = env.step(action)
        policy.update(state, action, reward, next_state)
        state = next_state
        total_reward += reward
    rewards.append(total_reward)

print("Average reward over last 50 episodes:", np.mean(rewards[-50:]))
print("Learned Value Function (Q-table):")
print(policy.Q)
